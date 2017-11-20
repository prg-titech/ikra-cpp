
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
  GLOBAL:448096
 Function _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I4BodyLm0EEEJEEEvPT_DpT0_:
  REG:4 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra3soa28storage_increase_size_kernelEPNS0_19StorageStrategyBaseEm:
  REG:10 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0_:
  REG:34 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:352 CONSTANT[2]:36 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0_:
  REG:27 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:352 CONSTANT[2]:36 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_22add_all_forces_to_thisEvEEE4callEPS4_mEUlS8_mE_JS8_mEEEvT_DpT0_:
  REG:42 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:344 CONSTANT[2]:64 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_11reset_forceEvEEE4callEPS4_mEUlS8_mE_JS8_mEEEvT_DpT0_:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:344 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z22codengen_simple_updateP4Bodyd:
  REG:34 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:36 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z18codegen_reset_massP4Body:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z24codegen_reset_force_halfP4Body:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z26codegen_reset_force_manualP4Body:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z14codegen_updateP4Bodyd:
  REG:27 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:36 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z17codegen_add_forceP4BodyS0_:
  REG:32 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:64 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z19codegen_reset_forceP4Body:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0

	code for sm_30
		Function : _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I4BodyLm0EEEJEEEvPT_DpT0_
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
        /*0048*/                   ST.E.64 [R2+0x8], RZ;                          /* 0x94000000202fdca5 */
        /*0050*/                   EXIT;                                          /* 0x8000000000001de7 */
        /*0058*/                   BRA 0x58;                                      /* 0x4003ffffe0001de7 */
        /*0060*/                   NOP;                                           /* 0x4000000000001de4 */
        /*0068*/                   NOP;                                           /* 0x4000000000001de4 */
        /*0070*/                   NOP;                                           /* 0x4000000000001de4 */
        /*0078*/                   NOP;                                           /* 0x4000000000001de4 */
		.....................................................................................................


		Function : _ZN4ikra3soa28storage_increase_size_kernelEPNS0_19StorageStrategyBaseEm
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                   /* 0x22c3f20042804307 */
        /*0008*/                   MOV R1, c[0x0][0x44];           /* 0x2800400110005de4 */
        /*0010*/                   MOV R2, c[0x0][0x140];          /* 0x2800400500009de4 */
        /*0018*/                   MOV R3, c[0x0][0x144];          /* 0x280040051000dde4 */
        /*0020*/                   LD.E.64 R4, [R2];               /* 0x8400000000211ca5 */
        /*0028*/                   MOV32I R8, 0x0;                 /* 0x1800000000021de2 */
        /*0030*/                   MOV32I R9, 0x0;                 /* 0x1800000000025de2 */
        /*0038*/                   IADD R6.CC, R4, c[0x0][0x148];  /* 0x4801400520419c03 */
                                                                   /* 0x20000002f042e287 */
        /*0048*/                   IADD.X R7, R5, c[0x0][0x14c];   /* 0x480040053051dc43 */
        /*0050*/                   ST.E.64 [R2], R6;               /* 0x9400000000219ca5 */
        /*0058*/                   ST.E.64 [R8], R4;               /* 0x9400000000811ca5 */
        /*0060*/                   EXIT;                           /* 0x8000000000001de7 */
        /*0068*/                   BRA 0x68;                       /* 0x4003ffffe0001de7 */
        /*0070*/                   NOP;                            /* 0x4000000000001de4 */
        /*0078*/                   NOP;                            /* 0x4000000000001de4 */
		..................................................................................


		Function : _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                              /* 0x22c2b202828232b7 */
        /*0008*/                   MOV R1, c[0x0][0x44];                                      /* 0x2800400110005de4 */
        /*0010*/                   S2R R0, SR_TID.X;                                          /* 0x2c00000084001c04 */
        /*0018*/                   S2R R3, SR_CTAID.X;                                        /* 0x2c0000009400dc04 */
        /*0020*/                   IMAD R0, R3, c[0x0][0x28], R0;                             /* 0x20004000a0301ca3 */
        /*0028*/                   ISUB RZ.CC, R0, c[0x0][0x150];                             /* 0x48014005400fdd03 */
        /*0030*/                   SHR R2, R0, 0x1f;                                          /* 0x5800c0007c009c23 */
        /*0038*/                   ISETP.GE.U32.X.AND P0, PT, R2, c[0x0][0x154], PT;          /* 0x1b0e40055021dc43 */
                                                                                              /* 0x2200423042c042e7 */
        /*0048*/               @P0 EXIT;                                                      /* 0x80000000000001e7 */
        /*0050*/                   IADD R4.CC, R0, c[0x0][0x148];                             /* 0x4801400520011c03 */
        /*0058*/                   MOV32I R0, 0x3;                                            /* 0x180000000c001de2 */
        /*0060*/                   IADD.X R2, R2, c[0x0][0x14c];                              /* 0x4800400530209c43 */
        /*0068*/                   SHL R3, R4, -0x1d;                                         /* 0x6000ffff8c40dc03 */
        /*0070*/                   IADD32I RZ.CC, R0, -0x20;                                  /* 0x0fffffff800fdc02 */
        /*0078*/                   SHR.U32 R5, R4, 0x1d;                                      /* 0x5800c00074415c03 */
                                                                                              /* 0x228042c042027237 */
        /*0088*/                   MOV32I R0, 0x0;                                            /* 0x1800000000001de2 */
        /*0090*/                   SHL R2, R2, 0x3;                                           /* 0x6000c0000c209c03 */
        /*0098*/                   CSETP.SFT.AND P0, PT, CC, PT;                              /* 0x040e00000151dc04 */
        /*00a0*/                   IMAD.U32.U32 R4.CC, R4, 0x8, R0;                           /* 0x2001c00020411c03 */
        /*00a8*/                   MOV32I R0, 0x0;                                            /* 0x1800000000001de2 */
        /*00b0*/               @P0 IADD R3, R2, R5;                                           /* 0x480000001420c003 */
        /*00b8*/                   MOV R2, R4;                                                /* 0x2800000010009de4 */
                                                                                              /* 0x223200423042b047 */
        /*00c8*/                   IADD.X R3, R0, R3;                                         /* 0x480000000c00dc43 */
        /*00d0*/                   MOV32I R0, 0xfffffc18;                                     /* 0x1bfffff060001de2 */
        /*00d8*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*00e0*/                   IADD32I R0, R0, 0x5;                                       /* 0x0800000014001c02 */
        /*00e8*/                   LD.E.64 R4, [R2+0x4e210];                                  /* 0x8400138840211ca5 */
        /*00f0*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*00f8*/                   SSY 0x1a0;                                                 /* 0x6000000280001c07 */
                                                                                              /* 0x22929252329283f7 */
        /*0108*/                   ISETP.NE.AND P0, PT, R0, RZ, PT;                           /* 0x1a8e0000fc01dc23 */
        /*0110*/                   MUFU.RCP64H R11, R7;                                       /* 0xc80000001872dc00 */
        /*0118*/                   DFMA R8, -R6, R10, c[0x2][0x0];                            /* 0x2014880000621e01 */
        /*0120*/                   DFMA R12, R8, R8, R8;                                      /* 0x2010000020831c01 */
        /*0128*/                   DMUL R8, R4, c[0x0][0x158];                                /* 0x5000400560421c01 */
        /*0130*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*0138*/                   DMUL R4, R8, R10;                                          /* 0x5000000028811c01 */
                                                                                              /* 0x2202b2c282820297 */
        /*0148*/                   DFMA R14, -R6, R4, R8;                                     /* 0x2010000010639e01 */
        /*0150*/                   DFMA R10, R10, R14, R4;                                    /* 0x2008000038a29c01 */
        /*0158*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*0160*/                   FFMA R4, RZ, R7, R11;                                      /* 0x301600001ff11c00 */
        /*0168*/                   FSETP.GT.AND P1, PT, |R4|, c[0x2][0xc], PT;                /* 0x220e48003043dc80 */
        /*0170*/               @P1 BRA CC.NEU, 0x198;                                         /* 0x40000000800005a7 */
        /*0178*/                   MOV R4, R6;                                                /* 0x2800000018011de4 */
                                                                                              /* 0x2230423042b2b2b7 */
        /*0188*/                   MOV R5, R7;                                                /* 0x280000001c015de4 */
        /*0190*/                   CAL 0xa70;                                                 /* 0x5000002360011c07 */
        /*0198*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*01a0*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*01a8*/                   MOV32I R8, 0x1;                                            /* 0x1800000004021de2 */
        /*01b0*/                   LD.E.64 R6, [R2+0x5dc10];                                  /* 0x8400177040219ca5 */
        /*01b8*/                   SSY 0x278;                                                 /* 0x60000002e0001c07 */
                                                                                              /* 0x229232929283f237 */
        /*01c8*/                   LD.E.64 R24, [R2+0x2ee10];                                 /* 0x84000bb840261ca5 */
        /*01d0*/                   LD.E.64 R22, [R2+0x3e810];                                 /* 0x84000fa040259ca5 */
        /*01d8*/                   MUFU.RCP64H R9, R5;                                        /* 0xc800000018525c00 */
        /*01e0*/                   DFMA R12, -R4, R8, c[0x2][0x0];                            /* 0x2010880000431e01 */
        /*01e8*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*01f0*/                   DFMA R12, R8, R12, R8;                                     /* 0x2010000030831c01 */
        /*01f8*/                   DMUL R8, R6, c[0x0][0x158];                                /* 0x5000400560621c01 */
                                                                                              /* 0x2272004292329297 */
        /*0208*/                   DMUL R6, R8, R12;                                          /* 0x5000000030819c01 */
        /*0210*/                   DFMA R16, -R4, R6, R8;                                     /* 0x2010000018441e01 */
        /*0218*/                   DFMA R16, R12, R16, R6;                                    /* 0x200c000040c41c01 */
        /*0220*/                   DADD R24, R10, R24;                                        /* 0x4800000060a61c01 */
        /*0228*/                   FFMA R6, RZ, R5, R17;                                      /* 0x3022000017f19c00 */
        /*0230*/                   ST.E.64 [R2+0x2ee10], R24;                                 /* 0x94000bb840261ca5 */
        /*0238*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
                                                                                              /* 0x2232b2b042b2b2c7 */
        /*0248*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003063dc80 */
        /*0250*/               @P1 BRA CC.NEU, 0x270;                                         /* 0x40000000600005a7 */
        /*0258*/                   CAL 0xa70;                                                 /* 0x5000002040011c07 */
        /*0260*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
        /*0268*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*0270*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0278*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
                                                                                              /* 0x22e2e04230423047 */
        /*0288*/                   LD.E.64 R12, [R2+0x4e210];                                 /* 0x8400138840231ca5 */
        /*0290*/                   DADD R16, R22, R16;                                        /* 0x4800000041641c01 */
        /*0298*/                   LD.E.64 R6, [R2+0xfa10];                                   /* 0x840003e840219ca5 */
        /*02a0*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*02a8*/                   LD.E.64 R8, [R2+0x1f410];                                  /* 0x840007d040221ca5 */
        /*02b0*/                   SSY 0x398;                                                 /* 0x6000000380001c07 */
        /*02b8*/                   ST.E.64 [R2+0x3e810], R16;                                 /* 0x94000fa040241ca5 */
                                                                                              /* 0x22929232929283f7 */
        /*02c8*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
        /*02d0*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*02d8*/                   DFMA R18, -R4, R14, c[0x2][0x0];                           /* 0x201c880000449e01 */
        /*02e0*/                   DFMA R18, R18, R18, R18;                                   /* 0x2024000049249c01 */
        /*02e8*/                   DFMA R18, R14, R18, R14;                                   /* 0x201c000048e49c01 */
        /*02f0*/                   DMUL R14, R12, c[0x0][0x158];                              /* 0x5000400560c39c01 */
        /*02f8*/                   DMUL R32, R14, R18;                                        /* 0x5000000048e81c01 */
                                                                                              /* 0x22d2004252323297 */
        /*0308*/                   DFMA R12, -R4, R32, R14;                                   /* 0x201c000080431e01 */
        /*0310*/                   DFMA R18, R18, R12, R32;                                   /* 0x2040000031249c01 */
        /*0318*/                   DFMA R6, R24, c[0x0][0x158], R6;                           /* 0x200c400561819c01 */
        /*0320*/                   DFMA R8, R10, c[0x0][0x158], R8;                           /* 0x2010400560a21c01 */
        /*0328*/                   FFMA R10, RZ, R5, R19;                                     /* 0x3026000017f29c00 */
        /*0330*/                   ST.E.64 [R2+0xfa10], R6;                                   /* 0x940003e840219ca5 */
        /*0338*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
                                                                                              /* 0x2202b2b042b2c047 */
        /*0348*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*0350*/                   ST.E.64 [R2+0x1f410], R8;                                  /* 0x940007d040221ca5 */
        /*0358*/               @P1 BRA CC.NEU, 0x390;                                         /* 0x40000000c00005a7 */
        /*0360*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0368*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
        /*0370*/                   CAL 0xa70;                                                 /* 0x5000001be0011c07 */
        /*0378*/                   MOV R18, R10;                                              /* 0x2800000028049de4 */
                                                                                              /* 0x223230423042b2b7 */
        /*0388*/                   MOV R19, R11;                                              /* 0x280000002c04dde4 */
        /*0390*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0398*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*03a0*/                   MOV32I R8, 0x1;                                            /* 0x1800000004021de2 */
        /*03a8*/                   LD.E.64 R6, [R2+0x5dc10];                                  /* 0x8400177040219ca5 */
        /*03b0*/                   SSY 0x460;                                                 /* 0x60000002a0001c07 */
        /*03b8*/                   LD.E.64 R24, [R2+0x2ee10];                                 /* 0x84000bb840261ca5 */
                                                                                              /* 0x22929232929283f7 */
        /*03c8*/                   LD.E.64 R22, [R2+0x3e810];                                 /* 0x84000fa040259ca5 */
        /*03d0*/                   MUFU.RCP64H R9, R5;                                        /* 0xc800000018525c00 */
        /*03d8*/                   DFMA R10, -R4, R8, c[0x2][0x0];                            /* 0x2010880000429e01 */
        /*03e0*/                   DFMA R10, R10, R10, R10;                                   /* 0x2014000028a29c01 */
        /*03e8*/                   DFMA R10, R8, R10, R8;                                     /* 0x2010000028829c01 */
        /*03f0*/                   DMUL R8, R6, c[0x0][0x158];                                /* 0x5000400560621c01 */
        /*03f8*/                   DMUL R6, R8, R10;                                          /* 0x5000000028819c01 */
                                                                                              /* 0x22c2720042923297 */
        /*0408*/                   DFMA R12, -R4, R6, R8;                                     /* 0x2010000018431e01 */
        /*0410*/                   DFMA R10, R10, R12, R6;                                    /* 0x200c000030a29c01 */
        /*0418*/                   DADD R24, R18, R24;                                        /* 0x4800000061261c01 */
        /*0420*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*0428*/                   ST.E.64 [R2+0x2ee10], R24;                                 /* 0x94000bb840261ca5 */
        /*0430*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*0438*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003063dc80 */
                                                                                              /* 0x2230423042b2b2b7 */
        /*0448*/               @P1 BRA CC.NEU, 0x458;                                         /* 0x40000000200005a7 */
        /*0450*/                   CAL 0xa70;                                                 /* 0x5000001860011c07 */
        /*0458*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0460*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*0468*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*0470*/                   LD.E.64 R12, [R2+0x4e210];                                 /* 0x8400138840231ca5 */
        /*0478*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
                                                                                              /* 0x229283f2e2e23047 */
        /*0488*/                   LD.E.64 R6, [R2+0xfa10];                                   /* 0x840003e840219ca5 */
        /*0490*/                   SSY 0x578;                                                 /* 0x6000000380001c07 */
        /*0498*/                   LD.E.64 R8, [R2+0x1f410];                                  /* 0x840007d040221ca5 */
        /*04a0*/                   ST.E.64 [R2+0x3e810], R22;                                 /* 0x94000fa040259ca5 */
        /*04a8*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
        /*04b0*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*04b8*/                   DFMA R16, -R4, R14, c[0x2][0x0];                           /* 0x201c880000441e01 */
                                                                                              /* 0x2232329292923297 */
        /*04c8*/                   DFMA R16, R16, R16, R16;                                   /* 0x2020000041041c01 */
        /*04d0*/                   DFMA R16, R14, R16, R14;                                   /* 0x201c000040e41c01 */
        /*04d8*/                   DMUL R14, R12, c[0x0][0x158];                              /* 0x5000400560c39c01 */
        /*04e0*/                   DMUL R18, R14, R16;                                        /* 0x5000000040e49c01 */
        /*04e8*/                   DFMA R12, -R4, R18, R14;                                   /* 0x201c000048431e01 */
        /*04f0*/                   DFMA R16, R16, R12, R18;                                   /* 0x2024000031041c01 */
        /*04f8*/                   DFMA R6, R24, c[0x0][0x158], R6;                           /* 0x200c400561819c01 */
                                                                                              /* 0x22b2c042d2004257 */
        /*0508*/                   DFMA R8, R10, c[0x0][0x158], R8;                           /* 0x2010400560a21c01 */
        /*0510*/                   FFMA R10, RZ, R5, R17;                                     /* 0x3022000017f29c00 */
        /*0518*/                   ST.E.64 [R2+0xfa10], R6;                                   /* 0x940003e840219ca5 */
        /*0520*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
        /*0528*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*0530*/                   ST.E.64 [R2+0x1f410], R8;                                  /* 0x940007d040221ca5 */
        /*0538*/               @P1 BRA CC.NEU, 0x570;                                         /* 0x40000000c00005a7 */
                                                                                              /* 0x2232b2b042b2b047 */
        /*0548*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0550*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
        /*0558*/                   CAL 0xa70;                                                 /* 0x5000001440011c07 */
        /*0560*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
        /*0568*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*0570*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0578*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
                                                                                              /* 0x229283f230423047 */
        /*0588*/                   LD.E.64 R8, [R2+0x5dc10];                                  /* 0x8400177040221ca5 */
        /*0590*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*0598*/                   LD.E.64 R6, [R2+0x2ee10];                                  /* 0x84000bb840219ca5 */
        /*05a0*/                   SSY 0x640;                                                 /* 0x6000000260001c07 */
        /*05a8*/                   LD.E.64 R22, [R2+0x3e810];                                 /* 0x84000fa040259ca5 */
        /*05b0*/                   MUFU.RCP64H R11, R5;                                       /* 0xc80000001852dc00 */
        /*05b8*/                   DFMA R12, -R4, R10, c[0x2][0x0];                           /* 0x2014880000431e01 */
                                                                                              /* 0x2292329292925237 */
        /*05c8*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*05d0*/                   DMUL R8, R8, c[0x0][0x158];                                /* 0x5000400560821c01 */
        /*05d8*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*05e0*/                   DMUL R14, R8, R10;                                         /* 0x5000000028839c01 */
        /*05e8*/                   DFMA R12, -R4, R14, R8;                                    /* 0x2010000038431e01 */
        /*05f0*/                   DFMA R10, R10, R12, R14;                                   /* 0x201c000030a29c01 */
        /*05f8*/                   DADD R24, R16, R6;                                         /* 0x4800000019061c01 */
                                                                                              /* 0x22b2b2b2c2720047 */
        /*0608*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*0610*/                   ST.E.64 [R2+0x2ee10], R24;                                 /* 0x94000bb840261ca5 */
        /*0618*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*0620*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003063dc80 */
        /*0628*/               @P1 BRA CC.NEU, 0x638;                                         /* 0x40000000200005a7 */
        /*0630*/                   CAL 0xa70;                                                 /* 0x50000010e0011c07 */
        /*0638*/                   NOP.S;                                                     /* 0x4000000000001df4 */
                                                                                              /* 0x22e2304230423047 */
        /*0648*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*0650*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*0658*/                   LD.E.64 R12, [R2+0x4e210];                                 /* 0x8400138840231ca5 */
        /*0660*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0668*/                   LD.E.64 R6, [R2+0xfa10];                                   /* 0x840003e840219ca5 */
        /*0670*/                   SSY 0x760;                                                 /* 0x60000003a0001c07 */
        /*0678*/                   LD.E.64 R8, [R2+0x1f410];                                  /* 0x840007d040221ca5 */
                                                                                              /* 0x229232929283f2e7 */
        /*0688*/                   ST.E.64 [R2+0x3e810], R22;                                 /* 0x94000fa040259ca5 */
        /*0690*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
        /*0698*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*06a0*/                   DFMA R16, -R4, R14, c[0x2][0x0];                           /* 0x201c880000441e01 */
        /*06a8*/                   DFMA R16, R16, R16, R16;                                   /* 0x2020000041041c01 */
        /*06b0*/                   DFMA R16, R14, R16, R14;                                   /* 0x201c000040e41c01 */
        /*06b8*/                   DMUL R14, R12, c[0x0][0x158];                              /* 0x5000400560c39c01 */
                                                                                              /* 0x2200425232329297 */
        /*06c8*/                   DMUL R18, R14, R16;                                        /* 0x5000000040e49c01 */
        /*06d0*/                   DFMA R12, -R4, R18, R14;                                   /* 0x201c000048431e01 */
        /*06d8*/                   DFMA R16, R16, R12, R18;                                   /* 0x2024000031041c01 */
        /*06e0*/                   DFMA R6, R24, c[0x0][0x158], R6;                           /* 0x200c400561819c01 */
        /*06e8*/                   DFMA R8, R10, c[0x0][0x158], R8;                           /* 0x2010400560a21c01 */
        /*06f0*/                   FFMA R10, RZ, R5, R17;                                     /* 0x3022000017f29c00 */
        /*06f8*/                   ST.E.64 [R2+0xfa10], R6;                                   /* 0x940003e840219ca5 */
                                                                                              /* 0x22b2b042b2c042d7 */
        /*0708*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
        /*0710*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*0718*/                   ST.E.64 [R2+0x1f410], R8;                                  /* 0x940007d040221ca5 */
        /*0720*/               @P1 BRA CC.NEU, 0x758;                                         /* 0x40000000c00005a7 */
        /*0728*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0730*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
        /*0738*/                   CAL 0xa70;                                                 /* 0x5000000cc0011c07 */
                                                                                              /* 0x2230423042b2b047 */
        /*0748*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
        /*0750*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*0758*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0760*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*0768*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*0770*/                   LD.E.64 R8, [R2+0x5dc10];                                  /* 0x8400177040221ca5 */
        /*0778*/                   SSY 0x828;                                                 /* 0x60000002a0001c07 */
                                                                                              /* 0x229252329283f237 */
        /*0788*/                   LD.E.64 R6, [R2+0x2ee10];                                  /* 0x84000bb840219ca5 */
        /*0790*/                   LD.E.64 R22, [R2+0x3e810];                                 /* 0x84000fa040259ca5 */
        /*0798*/                   MUFU.RCP64H R11, R5;                                       /* 0xc80000001852dc00 */
        /*07a0*/                   DFMA R12, -R4, R10, c[0x2][0x0];                           /* 0x2014880000431e01 */
        /*07a8*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*07b0*/                   DMUL R8, R8, c[0x0][0x158];                                /* 0x5000400560821c01 */
        /*07b8*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
                                                                                              /* 0x2272004292329297 */
        /*07c8*/                   DMUL R14, R8, R10;                                         /* 0x5000000028839c01 */
        /*07d0*/                   DFMA R12, -R4, R14, R8;                                    /* 0x2010000038431e01 */
        /*07d8*/                   DFMA R10, R10, R12, R14;                                   /* 0x201c000030a29c01 */
        /*07e0*/                   DADD R24, R16, R6;                                         /* 0x4800000019061c01 */
        /*07e8*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*07f0*/                   ST.E.64 [R2+0x2ee10], R24;                                 /* 0x94000bb840261ca5 */
        /*07f8*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
                                                                                              /* 0x22323042b2b2b2c7 */
        /*0808*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003063dc80 */
        /*0810*/               @P1 BRA CC.NEU, 0x820;                                         /* 0x40000000200005a7 */
        /*0818*/                   CAL 0xa70;                                                 /* 0x5000000940011c07 */
        /*0820*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0828*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*0830*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*0838*/                   LD.E.64 R12, [R2+0x4e210];                                 /* 0x8400138840231ca5 */
                                                                                              /* 0x2283f2e2e0423047 */
        /*0848*/                   LD.E.64 R6, [R2+0xfa10];                                   /* 0x840003e840219ca5 */
        /*0850*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0858*/                   LD.E.64 R8, [R2+0x1f410];                                  /* 0x840007d040221ca5 */
        /*0860*/                   SSY 0x940;                                                 /* 0x6000000360001c07 */
        /*0868*/                   ST.E.64 [R2+0x3e810], R22;                                 /* 0x94000fa040259ca5 */
        /*0870*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
        /*0878*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
                                                                                              /* 0x2232929292329297 */
        /*0888*/                   DFMA R16, -R4, R14, c[0x2][0x0];                           /* 0x201c880000441e01 */
        /*0890*/                   DFMA R16, R16, R16, R16;                                   /* 0x2020000041041c01 */
        /*0898*/                   DFMA R16, R14, R16, R14;                                   /* 0x201c000040e41c01 */
        /*08a0*/                   DMUL R14, R12, c[0x0][0x158];                              /* 0x5000400560c39c01 */
        /*08a8*/                   DMUL R18, R14, R16;                                        /* 0x5000000040e49c01 */
        /*08b0*/                   DFMA R12, -R4, R18, R14;                                   /* 0x201c000048431e01 */
        /*08b8*/                   DFMA R16, R16, R12, R18;                                   /* 0x2024000031041c01 */
                                                                                              /* 0x22c042d200425237 */
        /*08c8*/                   DFMA R6, R24, c[0x0][0x158], R6;                           /* 0x200c400561819c01 */
        /*08d0*/                   DFMA R8, R10, c[0x0][0x158], R8;                           /* 0x2010400560a21c01 */
        /*08d8*/                   FFMA R10, RZ, R5, R17;                                     /* 0x3022000017f29c00 */
        /*08e0*/                   ST.E.64 [R2+0xfa10], R6;                                   /* 0x940003e840219ca5 */
        /*08e8*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
        /*08f0*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*08f8*/                   ST.E.64 [R2+0x1f410], R8;                                  /* 0x940007d040221ca5 */
                                                                                              /* 0x22b2b042b2b042b7 */
        /*0908*/               @P1 BRA CC.NEU, 0x938;                                         /* 0x40000000a00005a7 */
        /*0910*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0918*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
        /*0920*/                   CAL 0xa70;                                                 /* 0x5000000520011c07 */
        /*0928*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
        /*0930*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*0938*/                   NOP.S;                                                     /* 0x4000000000001df4 */
                                                                                              /* 0x2283f23230423047 */
        /*0948*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*0950*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*0958*/                   LD.E.64 R8, [R2+0x5dc10];                                  /* 0x8400177040221ca5 */
        /*0960*/                   SSY 0xa10;                                                 /* 0x60000002a0001c07 */
        /*0968*/                   LD.E.64 R6, [R2+0x2ee10];                                  /* 0x84000bb840219ca5 */
        /*0970*/                   LD.E.64 R22, [R2+0x3e810];                                 /* 0x84000fa040259ca5 */
        /*0978*/                   MUFU.RCP64H R11, R5;                                       /* 0xc80000001852dc00 */
                                                                                              /* 0x2232929292523297 */
        /*0988*/                   DFMA R12, -R4, R10, c[0x2][0x0];                           /* 0x2014880000431e01 */
        /*0990*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*0998*/                   DMUL R8, R8, c[0x0][0x158];                                /* 0x5000400560821c01 */
        /*09a0*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*09a8*/                   DMUL R14, R8, R10;                                         /* 0x5000000028839c01 */
        /*09b0*/                   DFMA R12, -R4, R14, R8;                                    /* 0x2010000038431e01 */
        /*09b8*/                   DFMA R10, R10, R12, R14;                                   /* 0x201c000030a29c01 */
                                                                                              /* 0x22b2b2c272004297 */
        /*09c8*/                   DADD R24, R16, R6;                                         /* 0x4800000019061c01 */
        /*09d0*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*09d8*/                   ST.E.64 [R2+0x2ee10], R24;                                 /* 0x94000bb840261ca5 */
        /*09e0*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*09e8*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003063dc80 */
        /*09f0*/               @P1 BRA CC.NEU, 0xa00;                                         /* 0x40000000200005a7 */
        /*09f8*/                   CAL 0xa70;                                                 /* 0x50000001c0011c07 */
                                                                                              /* 0x2293f2e2e23202b7 */
        /*0a08*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0a10*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*0a18*/                   LD.E.64 R4, [R2+0xfa10];                                   /* 0x840003e840211ca5 */
        /*0a20*/                   LD.E.64 R6, [R2+0x1f410];                                  /* 0x840007d040219ca5 */
        /*0a28*/                   ST.E.64 [R2+0x3e810], R22;                                 /* 0x94000fa040259ca5 */
        /*0a30*/                   LD.E.64 R8, [R2+0x3e810];                                  /* 0x84000fa040221ca5 */
        /*0a38*/                   DFMA R4, R24, c[0x0][0x158], R4;                           /* 0x2008400561811c01 */
                                                                                              /* 0x220042e2e0429377 */
        /*0a48*/                   ST.E.64 [R2+0xfa10], R4;                                   /* 0x940003e840211ca5 */
        /*0a50*/                   DFMA R6, R8, c[0x0][0x158], R6;                            /* 0x200c400560819c01 */
        /*0a58*/                   ST.E.64 [R2+0x1f410], R6;                                  /* 0x940007d040219ca5 */
        /*0a60*/               @P0 BRA 0xd8;                                                  /* 0x4003ffd9c00001e7 */
        /*0a68*/                   EXIT;                                                      /* 0x8000000000001de7 */
        /*0a70*/                   LOP32I.AND R6, R5, 0x40000000;                             /* 0x3900000000519c02 */
        /*0a78*/                   LOP32I.AND R13, R9, 0x7f800000;                            /* 0x39fe000000935c02 */
                                                                                              /* 0x22028042c0427047 */
        /*0a88*/                   MOV32I R12, 0x1ff00000;                                    /* 0x187fc00000031de2 */
        /*0a90*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0a98*/                   ISETP.LT.U32.AND P1, PT, R6, c[0x2][0x10], PT;             /* 0x188e48004063dc03 */
        /*0aa0*/                   MOV R6, RZ;                                                /* 0x28000000fc019de4 */
        /*0aa8*/                   SEL R7, R12, c[0x2][0x14], !P1;                            /* 0x2012480050c1dc04 */
        /*0ab0*/                   PBK 0xd58;                                                 /* 0x6800000a80001c07 */
        /*0ab8*/                   DMUL R10, R4, R6;                                          /* 0x5000000018429c01 */
                                                                                              /* 0x225232928042c047 */
        /*0ac8*/                   ISETP.LT.U32.AND P1, PT, R13, c[0x2][0x18], PT;            /* 0x188e480060d3dc03 */
        /*0ad0*/                   MOV R18, RZ;                                               /* 0x28000000fc049de4 */
        /*0ad8*/                   MUFU.RCP64H R15, R11;                                      /* 0xc800000018b3dc00 */
        /*0ae0*/                   SEL R19, R12, c[0x2][0x14], !P1;                           /* 0x2012480050c4dc04 */
        /*0ae8*/                   DFMA R20, -R10, R14, c[0x2][0x0];                          /* 0x201c880000a51e01 */
        /*0af0*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*0af8*/                   DMUL R16, R8, R18;                                         /* 0x5000000048841c01 */
                                                                                              /* 0x2202b2d292929297 */
        /*0b08*/                   DFMA R14, R14, R20, R14;                                   /* 0x201c000050e39c01 */
        /*0b10*/                   DMUL R20, R16, R14;                                        /* 0x5000000039051c01 */
        /*0b18*/                   DFMA R10, -R10, R20, R16;                                  /* 0x2020000050a29e01 */
        /*0b20*/                   DFMA R10, R14, R10, R20;                                   /* 0x2028000028e29c01 */
        /*0b28*/                   DSETP.LEU.AND P1, PT, |R10|, RZ, PT;                       /* 0x1d8e0000fca3dc81 */
        /*0b30*/               @P1 BRA 0xcd8;                                                 /* 0x40000006800005e7 */
        /*0b38*/                   ISETP.GT.U32.AND P1, PT, R13, c[0x2][0x1c], PT;            /* 0x1a0e480070d3dc03 */
                                                                                              /* 0x22523252328202b7 */
        /*0b48*/                   DMUL R14, R6, R10;                                         /* 0x5000000028639c01 */
        /*0b50*/                   SEL R13, R12, c[0x2][0x14], !P1;                           /* 0x2012480050c35c04 */
        /*0b58*/                   MOV R12, RZ;                                               /* 0x28000000fc031de4 */
        /*0b60*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0b68*/                   DMUL R10, R12, R14;                                        /* 0x5000000038c29c01 */
        /*0b70*/                   DMUL R6, R6, R16;                                          /* 0x5000000040619c01 */
        /*0b78*/                   DFMA R12, R4, R10, -R8;                                    /* 0x2010000028431d01 */
                                                                                              /* 0x2202b2c27202d297 */
        /*0b88*/                   DFMA R14, R4, R6, -R8;                                     /* 0x2010000018439d01 */
        /*0b90*/                   DSETP.GT.AND P1, PT, |R12|, |R14|, PT;                     /* 0x1a0e000038c3dcc1 */
        /*0b98*/                   SEL R7, R7, R11, P1;                                       /* 0x200200002c71dc04 */
        /*0ba0*/                   SEL R6, R6, R10, P1;                                       /* 0x2002000028619c04 */
        /*0ba8*/                   FSETP.GTU.AND P1, PT, |R7|, 1.469367938527859385e-39, PT;  /* 0x260ec0040073dc80 */
        /*0bb0*/               @P1 BRK;                                                       /* 0xa8000000000005e7 */
        /*0bb8*/                   FSETP.LT.AND P1, PT, |R9|, 1.5046327690525280102e-36, PT;  /* 0x208ec1000093dc80 */
                                                                                              /* 0x2232829200420047 */
        /*0bc8*/                   MOV32I R15, 0x3ff00000;                                    /* 0x18ffc0000003dde2 */
        /*0bd0*/                   LOP32I.AND R10, R6, 0xfffffffe;                            /* 0x3bfffffff8629c02 */
        /*0bd8*/                   MOV R14, RZ;                                               /* 0x28000000fc039de4 */
        /*0be0*/                   MOV R11, R7;                                               /* 0x280000001c02dde4 */
        /*0be8*/                   LOP32I.OR R6, R6, 0x1;                                     /* 0x3800000004619c42 */
        /*0bf0*/                   SEL R15, R15, c[0x2][0x20], !P1;                           /* 0x2012480080f3dc04 */
        /*0bf8*/                   DMUL R4, R4, R14;                                          /* 0x5000000038411c01 */
                                                                                              /* 0x22c04282d2923297 */
        /*0c08*/                   DMUL R8, R8, R14;                                          /* 0x5000000038821c01 */
        /*0c10*/                   DFMA R12, R10, R4, -R8;                                    /* 0x2010000010a31d01 */
        /*0c18*/                   DFMA R14, R6, R4, -R8;                                     /* 0x2010000010639d01 */
        /*0c20*/                   DSETP.GT.AND P1, PT, |R12|, |R14|, PT;                     /* 0x1a0e000038c3dcc1 */
        /*0c28*/                   SEL R11, R6, R10, P1;                                      /* 0x200200002862dc04 */
        /*0c30*/                   IADD32I R13.CC, R11, 0x1;                                  /* 0x0c00000004b35c02 */
        /*0c38*/                   LOP32I.AND R10, R11, 0x1;                                  /* 0x3800000004b29c02 */
                                                                                              /* 0x2272004202b20047 */
        /*0c48*/                   ISETP.EQ.U32.AND P1, PT, R10, 0x1, PT;                     /* 0x190ec00004a3dc03 */
        /*0c50*/                   IADD.X R12, RZ, R7;                                        /* 0x480000001ff31c43 */
        /*0c58*/                   IADD32I R6.CC, R11, -0x1;                                  /* 0x0ffffffffcb19c02 */
        /*0c60*/                   SEL R13, R11, R13, !P1;                                    /* 0x2012000034b35c04 */
        /*0c68*/                   IADD32I.X R10, R7, -0x1;                                   /* 0x0bfffffffc729c42 */
        /*0c70*/                   SEL R12, R7, R12, !P1;                                     /* 0x2012000030731c04 */
        /*0c78*/                   SEL R6, R6, R11, !P1;                                      /* 0x201200002c619c04 */
                                                                                              /* 0x2202d29232820207 */
        /*0c88*/                   SEL R7, R10, R7, !P1;                                      /* 0x201200001ca1dc04 */
        /*0c90*/                   MOV R10, R13;                                              /* 0x2800000034029de4 */
        /*0c98*/                   MOV R11, R12;                                              /* 0x280000003002dde4 */
        /*0ca0*/                   DFMA R10, R4, R10, -R8;                                    /* 0x2010000028429d01 */
        /*0ca8*/                   DFMA R4, R4, R6, -R8;                                      /* 0x2010000018411d01 */
        /*0cb0*/                   DSETP.GT.AND P1, PT, |R10|, |R4|, PT;                      /* 0x1a0e000010a3dcc1 */
        /*0cb8*/                   SEL R6, R6, R13, P1;                                       /* 0x2002000034619c04 */
                                                                                              /* 0x22d28042b2d2e047 */
        /*0cc8*/                   SEL R7, R7, R12, P1;                                       /* 0x200200003071dc04 */
        /*0cd0*/                   BRK;                                                       /* 0xa800000000001de7 */
        /*0cd8*/                   DSETP.EQ.AND P1, PT, R10, RZ, PT;                          /* 0x190e0000fca3dc01 */
        /*0ce0*/               @P1 BRA 0xd40;                                                 /* 0x40000001600005e7 */
        /*0ce8*/                   MUFU.RCP64H R7, R5;                                        /* 0xc80000001851dc00 */
        /*0cf0*/                   MOV R6, RZ;                                                /* 0x28000000fc019de4 */
        /*0cf8*/                   DSETP.GT.AND P2, PT, |R6|, RZ, PT;                         /* 0x1a0e0000fc65dc81 */
                                                                                              /* 0x22e20282027202d7 */
        /*0d08*/              @!P2 DSETP.NEU.AND P1, PT, |R4|, +INF , PT;                     /* 0x1e8edffc0043e881 */
        /*0d10*/              @!P2 SEL R4, R4, R6, P1;                                        /* 0x2002000018412804 */
        /*0d18*/              @!P2 SEL R5, R5, R7, P1;                                        /* 0x200200001c516804 */
        /*0d20*/              @!P2 MOV R6, R4;                                                /* 0x280000001001a9e4 */
        /*0d28*/              @!P2 MOV R7, R5;                                                /* 0x280000001401e9e4 */
        /*0d30*/                   DMUL R6, R8, R6;                                           /* 0x5000000018819c01 */
        /*0d38*/                   BRK;                                                       /* 0xa800000000001de7 */
                                                                                              /* 0x200002e20042b207 */
        /*0d48*/                   DMUL R6, R8, R4;                                           /* 0x5000000010819c01 */
        /*0d50*/                   BRK;                                                       /* 0xa800000000001de7 */
        /*0d58*/                   MOV R10, R6;                                               /* 0x2800000018029de4 */
        /*0d60*/                   MOV R11, R7;                                               /* 0x280000001c02dde4 */
        /*0d68*/                   RET;                                                       /* 0x9000000000001de7 */
        /*0d70*/                   BRA 0xd70;                                                 /* 0x4003ffffe0001de7 */
        /*0d78*/                   NOP;                                                       /* 0x4000000000001de4 */
		........................................................................................................................................................................


		Function : _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                               /* 0x22c2b202828232b7 */
        /*0008*/                   MOV R1, c[0x0][0x44];                                       /* 0x2800400110005de4 */
        /*0010*/                   S2R R0, SR_TID.X;                                           /* 0x2c00000084001c04 */
        /*0018*/                   S2R R3, SR_CTAID.X;                                         /* 0x2c0000009400dc04 */
        /*0020*/                   IMAD R0, R3, c[0x0][0x28], R0;                              /* 0x20004000a0301ca3 */
        /*0028*/                   ISUB RZ.CC, R0, c[0x0][0x150];                              /* 0x48014005400fdd03 */
        /*0030*/                   SHR R2, R0, 0x1f;                                           /* 0x5800c0007c009c23 */
        /*0038*/                   ISETP.GE.U32.X.AND P0, PT, R2, c[0x0][0x154], PT;           /* 0x1b0e40055021dc43 */
                                                                                               /* 0x2200420042c042e7 */
        /*0048*/               @P0 EXIT;                                                       /* 0x80000000000001e7 */
        /*0050*/                   IADD R4.CC, R0, c[0x0][0x148];                              /* 0x4801400520011c03 */
        /*0058*/                   MOV32I R0, 0x3;                                             /* 0x180000000c001de2 */
        /*0060*/                   IADD.X R2, R2, c[0x0][0x14c];                               /* 0x4800400530209c43 */
        /*0068*/                   SHL R3, R4, -0x1d;                                          /* 0x6000ffff8c40dc03 */
        /*0070*/                   IADD32I RZ.CC, R0, -0x20;                                   /* 0x0fffffff800fdc02 */
        /*0078*/                   MOV32I R8, 0x1;                                             /* 0x1800000004021de2 */
                                                                                               /* 0x22b0420242421047 */
        /*0088*/                   MOV32I R0, 0x0;                                             /* 0x1800000000001de2 */
        /*0090*/                   SSY 0x190;                                                  /* 0x60000003e0001c07 */
        /*0098*/                   SHR.U32 R5, R4, 0x1d;                                       /* 0x5800c00074415c03 */
        /*00a0*/                   SHL R2, R2, 0x3;                                            /* 0x6000c0000c209c03 */
        /*00a8*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*00b0*/                   IMAD.U32.U32 R6.CC, R4, 0x8, R0;                            /* 0x2001c00020419c03 */
        /*00b8*/                   MOV32I R0, 0x0;                                             /* 0x1800000000001de2 */
                                                                                               /* 0x22329283f2328287 */
        /*00c8*/               @P0 IADD R3, R2, R5;                                            /* 0x480000001420c003 */
        /*00d0*/                   IADD.X R7, R0, R3;                                          /* 0x480000000c01dc43 */
        /*00d8*/                   LD.E.64 R4, [R6+0x10];                                      /* 0x8400000040611ca5 */
        /*00e0*/                   LD.E.64 R2, [R6+0x4e210];                                   /* 0x8400138840609ca5 */
        /*00e8*/                   MUFU.RCP64H R9, R5;                                         /* 0xc800000018525c00 */
        /*00f0*/                   DFMA R10, -R4, R8, c[0x2][0x0];                             /* 0x2010880000429e01 */
        /*00f8*/                   DFMA R10, R10, R10, R10;                                    /* 0x2014000028a29c01 */
                                                                                               /* 0x2282820292929257 */
        /*0108*/                   DMUL R14, R2, c[0x0][0x158];                                /* 0x5000400560239c01 */
        /*0110*/                   DFMA R8, R8, R10, R8;                                       /* 0x2010000028821c01 */
        /*0118*/                   DMUL R2, R14, R8;                                           /* 0x5000000020e09c01 */
        /*0120*/                   DFMA R12, -R4, R2, R14;                                     /* 0x201c000008431e01 */
        /*0128*/                   DFMA R8, R8, R12, R2;                                       /* 0x2004000030821c01 */
        /*0130*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                 /* 0x160f480020ffdc80 */
        /*0138*/                   FFMA R0, RZ, R5, R9;                                        /* 0x3012000017f01c00 */
                                                                                               /* 0x22b042b2b042b2c7 */
        /*0148*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0xc], PT;                 /* 0x220e48003001dc80 */
        /*0150*/               @P0 BRA CC.NEU, 0x180;                                          /* 0x40000000a00001a7 */
        /*0158*/                   MOV R10, R4;                                                /* 0x2800000010029de4 */
        /*0160*/                   MOV R11, R5;                                                /* 0x280000001402dde4 */
        /*0168*/                   CAL 0x320;                                                  /* 0x50000006c0011c07 */
        /*0170*/                   MOV R8, R22;                                                /* 0x2800000058021de4 */
        /*0178*/                   MOV R9, R23;                                                /* 0x280000005c025de4 */
                                                                                               /* 0x22323230423042b7 */
        /*0188*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0190*/                   LD.E.64 R10, [R6+0x10];                                     /* 0x8400000040629ca5 */
        /*0198*/                   MOV32I R16, 0x1;                                            /* 0x1800000004041de2 */
        /*01a0*/                   LD.E.64 R14, [R6+0x5dc10];                                  /* 0x8400177040639ca5 */
        /*01a8*/                   SSY 0x278;                                                  /* 0x6000000320001c07 */
        /*01b0*/                   LD.E.64 R18, [R6+0x2ee10];                                  /* 0x84000bb840649ca5 */
        /*01b8*/                   LD.E.64 R12, [R6+0xfa10];                                   /* 0x840003e840631ca5 */
                                                                                               /* 0x229252329283f237 */
        /*01c8*/                   LD.E.64 R4, [R6+0x3e810];                                   /* 0x84000fa040611ca5 */
        /*01d0*/                   LD.E.64 R2, [R6+0x1f410];                                   /* 0x840007d040609ca5 */
        /*01d8*/                   MUFU.RCP64H R17, R11;                                       /* 0xc800000018b45c00 */
        /*01e0*/                   DFMA R20, -R10, R16, c[0x2][0x0];                           /* 0x2020880000a51e01 */
        /*01e8*/                   DFMA R20, R20, R20, R20;                                    /* 0x2028000051451c01 */
        /*01f0*/                   DMUL R14, R14, c[0x0][0x158];                               /* 0x5000400560e39c01 */
        /*01f8*/                   DFMA R16, R16, R20, R16;                                    /* 0x2020000051041c01 */
                                                                                               /* 0x2200425232523297 */
        /*0208*/                   DMUL R20, R14, R16;                                         /* 0x5000000040e51c01 */
        /*0210*/                   DFMA R22, -R10, R20, R14;                                   /* 0x201c000050a59e01 */
        /*0218*/                   DADD R18, R8, R18;                                          /* 0x4800000048849c01 */
        /*0220*/                   DFMA R22, R16, R22, R20;                                    /* 0x2028000059059c01 */
        /*0228*/                   DFMA R8, R18, c[0x0][0x158], R12;                           /* 0x2018400561221c01 */
        /*0230*/                   FFMA R0, RZ, R11, R23;                                      /* 0x302e00002ff01c00 */
        /*0238*/                   ST.E.64 [R6+0x2ee10], R18;                                  /* 0x94000bb840649ca5 */
                                                                                               /* 0x2232b2b2b2c042d7 */
        /*0248*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                 /* 0x160f480020ffdc80 */
        /*0250*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0xc], PT;                 /* 0x220e48003001dc80 */
        /*0258*/                   ST.E.64 [R6+0xfa10], R8;                                    /* 0x940003e840621ca5 */
        /*0260*/               @P0 BRA CC.NEU, 0x270;                                          /* 0x40000000200001a7 */
        /*0268*/                   CAL 0x320;                                                  /* 0x50000002c0011c07 */
        /*0270*/                   DADD.S R4, R4, R22;                                         /* 0x4800000058411c11 */
        /*0278*/                   DSETP.GT.AND P0, PT, R8, 1, PT;                             /* 0x1a0ecffc0081dc01 */
                                                                                               /* 0x2232b042d2804207 */
        /*0288*/                   DFMA R2, R4, c[0x0][0x158], R2;                             /* 0x2004400560409c01 */
        /*0290*/                   ST.E.64 [R6+0x3e810], R4;                                   /* 0x94000fa040611ca5 */
        /*0298*/                   SSY 0x2e0;                                                  /* 0x6000000100001c07 */
        /*02a0*/                   DSETP.LT.OR P0, PT, R8, -1, P0;                             /* 0x18a0effc0081dc01 */
        /*02a8*/                   ST.E.64 [R6+0x1f410], R2;                                   /* 0x940007d040609ca5 */
        /*02b0*/              @!P0 NOP.S;                                                      /* 0x40000000000021f4 */
        /*02b8*/                   LD.E.64 R4, [R6+0x2ee10];                                   /* 0x84000bb840611ca5 */
                                                                                               /* 0x23f2e2d2d2b283f7 */
        /*02c8*/                   LD.E.64 R2, [R6+0x1f410];                                   /* 0x840007d040609ca5 */
        /*02d0*/                   F2F.F64.F64 R4, -R4;                                        /* 0x1000000011b11d04 */
        /*02d8*/                   ST.E.64.S [R6+0x2ee10], R4;                                 /* 0x94000bb840611cb5 */
        /*02e0*/                   DSETP.GT.AND P0, PT, R2, 1, PT;                             /* 0x1a0ecffc0021dc01 */
        /*02e8*/                   DSETP.LT.OR P0, PT, R2, -1, P0;                             /* 0x18a0effc0021dc01 */
        /*02f0*/              @!P0 EXIT;                                                       /* 0x80000000000021e7 */
        /*02f8*/                   LD.E.64 R2, [R6+0x3e810];                                   /* 0x84000fa040609ca5 */
                                                                                               /* 0x2200428042e04287 */
        /*0308*/                   F2F.F64.F64 R2, -R2;                                        /* 0x1000000009b09d04 */
        /*0310*/                   ST.E.64 [R6+0x3e810], R2;                                   /* 0x94000fa040609ca5 */
        /*0318*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*0320*/                   LOP32I.AND R0, R11, 0x40000000;                             /* 0x3900000000b01c02 */
        /*0328*/                   LOP32I.AND R26, R15, 0x7f800000;                            /* 0x39fe000000f69c02 */
        /*0330*/                   ISETP.LT.U32.AND P0, PT, R0, c[0x2][0x10], PT;              /* 0x188e48004001dc03 */
        /*0338*/                   MOV32I R0, 0x1ff00000;                                      /* 0x187fc00000001de2 */
                                                                                               /* 0x228282028042b047 */
        /*0348*/                   MOV R12, RZ;                                                /* 0x28000000fc031de4 */
        /*0350*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
        /*0358*/                   SEL R13, R0, c[0x2][0x14], !P0;                             /* 0x2010480050035c04 */
        /*0360*/                   MOV R24, RZ;                                                /* 0x28000000fc061de4 */
        /*0368*/                   DMUL R16, R10, R12;                                         /* 0x5000000030a41c01 */
        /*0370*/                   ISETP.LT.U32.AND P0, PT, R26, c[0x2][0x18], PT;             /* 0x188e480061a1dc03 */
        /*0378*/                   MUFU.RCP64H R19, R17;                                       /* 0xc80000001914dc00 */
                                                                                               /* 0x2292929252328207 */
        /*0388*/                   DFMA R20, -R16, R18, c[0x2][0x0];                           /* 0x2024880001051e01 */
        /*0390*/                   SEL R25, R0, c[0x2][0x14], !P0;                             /* 0x2010480050065c04 */
        /*0398*/                   DFMA R22, R20, R20, R20;                                    /* 0x2028000051459c01 */
        /*03a0*/                   DMUL R20, R14, R24;                                         /* 0x5000000060e51c01 */
        /*03a8*/                   DFMA R18, R18, R22, R18;                                    /* 0x2024000059249c01 */
        /*03b0*/                   DMUL R22, R20, R18;                                         /* 0x5000000049459c01 */
        /*03b8*/                   DFMA R16, -R16, R22, R20;                                   /* 0x2028000059041e01 */
                                                                                               /* 0x2282b20042b2d297 */
        /*03c8*/                   DFMA R22, R18, R16, R22;                                    /* 0x202c000041259c01 */
        /*03d0*/                   DSETP.LEU.AND P0, PT, |R22|, RZ, PT;                        /* 0x1d8e0000fd61dc81 */
        /*03d8*/               @P0 BRA 0x590;                                                  /* 0x40000006c00001e7 */
        /*03e0*/                   ISETP.GT.U32.AND P0, PT, R26, c[0x2][0x1c], PT;             /* 0x1a0e480071a1dc03 */
        /*03e8*/                   DMUL R18, R12, R22;                                         /* 0x5000000058c49c01 */
        /*03f0*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*03f8*/                   SEL R17, R0, c[0x2][0x14], !P0;                             /* 0x2010480050045c04 */
                                                                                               /* 0x2202d29252325237 */
        /*0408*/                   DMUL R22, R22, R16;                                         /* 0x5000000041659c01 */
        /*0410*/                   DMUL R16, R16, R18;                                         /* 0x5000000049041c01 */
        /*0418*/                   DMUL R12, R12, R22;                                         /* 0x5000000058c31c01 */
        /*0420*/                   DFMA R18, R10, R16, -R14;                                   /* 0x201c000040a49d01 */
        /*0428*/                   DFMA R20, R10, R12, -R14;                                   /* 0x201c000030a51d01 */
        /*0430*/                   DSETP.GT.AND P0, PT, |R18|, |R20|, PT;                      /* 0x1a0e00005121dcc1 */
        /*0438*/                   SEL R13, R13, R17, P0;                                      /* 0x2000000044d35c04 */
                                                                                               /* 0x2270420042b2c277 */
        /*0448*/                   SEL R12, R12, R16, P0;                                      /* 0x2000000040c31c04 */
        /*0450*/                   FSETP.GTU.AND P0, PT, |R13|, 1.469367938527859385e-39, PT;  /* 0x260ec00400d1dc80 */
        /*0458*/               @P0 BRA 0x600;                                                  /* 0x40000006800001e7 */
        /*0460*/                   FSETP.LT.AND P0, PT, |R15|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000f1dc80 */
        /*0468*/                   LOP32I.AND R0, R12, 0xfffffffe;                             /* 0x3bfffffff8c01c02 */
        /*0470*/                   MOV32I R17, 0x3ff00000;                                     /* 0x18ffc00000045de2 */
        /*0478*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
                                                                                               /* 0x2280420232823047 */
        /*0488*/                   LOP32I.OR R20, R12, 0x1;                                    /* 0x3800000004c51c42 */
        /*0490*/                   MOV R12, R0;                                                /* 0x2800000000031de4 */
        /*0498*/                   SEL R17, R17, c[0x2][0x20], !P0;                            /* 0x2010480081145c04 */
        /*04a0*/                   DMUL R10, R10, R16;                                         /* 0x5000000040a29c01 */
        /*04a8*/                   DMUL R14, R14, R16;                                         /* 0x5000000040e39c01 */
        /*04b0*/                   MOV R16, R20;                                               /* 0x2800000050041de4 */
        /*04b8*/                   MOV R17, R13;                                               /* 0x2800000034045de4 */
                                                                                               /* 0x2232804282d29237 */
        /*04c8*/                   DFMA R18, R12, R10, -R14;                                   /* 0x201c000028c49d01 */
        /*04d0*/                   DFMA R16, R16, R10, -R14;                                   /* 0x201c000029041d01 */
        /*04d8*/                   DSETP.GT.AND P0, PT, |R18|, |R16|, PT;                      /* 0x1a0e00004121dcc1 */
        /*04e0*/                   SEL R20, R20, R0, P0;                                       /* 0x2000000001451c04 */
        /*04e8*/                   IADD32I R19.CC, R20, 0x1;                                   /* 0x0c0000000544dc02 */
        /*04f0*/                   LOP32I.AND R12, R20, 0x1;                                   /* 0x3800000005431c02 */
        /*04f8*/                   ISETP.EQ.U32.AND P0, PT, R12, 0x1, PT;                      /* 0x190ec00004c1dc03 */
                                                                                               /* 0x220262020042c047 */
        /*0508*/                   IADD.X R0, RZ, R13;                                         /* 0x4800000037f01c43 */
        /*0510*/                   IADD32I R16.CC, R20, -0x1;                                  /* 0x0ffffffffd441c02 */
        /*0518*/                   SEL R19, R20, R19, !P0;                                     /* 0x201000004d44dc04 */
        /*0520*/                   IADD32I.X R17, R13, -0x1;                                   /* 0x0bfffffffcd45c42 */
        /*0528*/                   SEL R0, R13, R0, !P0;                                       /* 0x2010000000d01c04 */
        /*0530*/                   SEL R16, R16, R20, !P0;                                     /* 0x2010000051041c04 */
        /*0538*/                   SEL R17, R17, R13, !P0;                                     /* 0x2010000035145c04 */
                                                                                               /* 0x220202d292328207 */
        /*0548*/                   MOV R12, R19;                                               /* 0x280000004c031de4 */
        /*0550*/                   MOV R13, R0;                                                /* 0x2800000000035de4 */
        /*0558*/                   DFMA R12, R10, R12, -R14;                                   /* 0x201c000030a31d01 */
        /*0560*/                   DFMA R10, R10, R16, -R14;                                   /* 0x201c000040a29d01 */
        /*0568*/                   DSETP.GT.AND P0, PT, |R12|, |R10|, PT;                      /* 0x1a0e000028c1dcc1 */
        /*0570*/                   SEL R12, R16, R19, P0;                                      /* 0x200000004d031c04 */
        /*0578*/                   SEL R13, R17, R0, P0;                                       /* 0x2000000001135c04 */
                                                                                               /* 0x22d2d28042b2d2e7 */
        /*0588*/                   BRA 0x600;                                                  /* 0x40000001c0001de7 */
        /*0590*/                   DSETP.EQ.AND P0, PT, R22, RZ, PT;                           /* 0x190e0000fd61dc01 */
        /*0598*/               @P0 BRA 0x5f8;                                                  /* 0x40000001600001e7 */
        /*05a0*/                   MUFU.RCP64H R13, R11;                                       /* 0xc800000018b35c00 */
        /*05a8*/                   MOV R12, RZ;                                                /* 0x28000000fc031de4 */
        /*05b0*/                   DSETP.GT.AND P1, PT, |R12|, RZ, PT;                         /* 0x1a0e0000fcc3dc81 */
        /*05b8*/              @!P1 DSETP.NEU.AND P0, PT, |R10|, +INF , PT;                     /* 0x1e8edffc00a1e481 */
                                                                                               /* 0x22b2e20282027207 */
        /*05c8*/              @!P1 SEL R0, R10, R12, P0;                                       /* 0x2000000030a02404 */
        /*05d0*/              @!P1 SEL R11, R11, R13, P0;                                      /* 0x2000000034b2e404 */
        /*05d8*/              @!P1 MOV R12, R0;                                                /* 0x28000000000325e4 */
        /*05e0*/              @!P1 MOV R13, R11;                                               /* 0x280000002c0365e4 */
        /*05e8*/                   DMUL R12, R14, R12;                                         /* 0x5000000030e31c01 */
        /*05f0*/                   BRA 0x600;                                                  /* 0x4000000020001de7 */
        /*05f8*/                   DMUL R12, R14, R10;                                         /* 0x5000000028e31c01 */
                                                                                               /* 0x2000000002e20047 */
        /*0608*/                   MOV R22, R12;                                               /* 0x2800000030059de4 */
        /*0610*/                   MOV R23, R13;                                               /* 0x280000003405dde4 */
        /*0618*/                   RET;                                                        /* 0x9000000000001de7 */
        /*0620*/                   BRA 0x620;                                                  /* 0x4003ffffe0001de7 */
        /*0628*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*0630*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*0638*/                   NOP;                                                        /* 0x4000000000001de4 */
		.......................................................................................................................................................


		Function : _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_22add_all_forces_to_thisEvEEE4callEPS4_mEUlS8_mE_JS8_mEEEvT_DpT0_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                               /* 0x22c2b202828232b7 */
        /*0008*/                   MOV R1, c[0x0][0x44];                                       /* 0x2800400110005de4 */
        /*0010*/                   S2R R16, SR_TID.X;                                          /* 0x2c00000084041c04 */
        /*0018*/                   S2R R3, SR_CTAID.X;                                         /* 0x2c0000009400dc04 */
        /*0020*/                   IMAD R16, R3, c[0x0][0x28], R16;                            /* 0x20204000a0341ca3 */
        /*0028*/                   ISUB RZ.CC, R16, c[0x0][0x150];                             /* 0x48014005410fdd03 */
        /*0030*/                   SHR R0, R16, 0x1f;                                          /* 0x5800c0007d001c23 */
        /*0038*/                   ISETP.GE.U32.X.AND P0, PT, R0, c[0x0][0x154], PT;           /* 0x1b0e40055001dc43 */
                                                                                               /* 0x22004220420042e7 */
        /*0048*/               @P0 EXIT;                                                       /* 0x80000000000001e7 */
        /*0050*/                   IADD R16.CC, R16, c[0x0][0x148];                            /* 0x4801400521041c03 */
        /*0058*/                   MOV32I R39, 0x1f40;                                         /* 0x1800007d0009dde2 */
        /*0060*/                   MOV32I R4, 0x0;                                             /* 0x1800000000011de2 */
        /*0068*/                   MOV R40, RZ;                                                /* 0x28000000fc0a1de4 */
        /*0070*/                   MOV32I R5, 0x0;                                             /* 0x1800000000015de2 */
        /*0078*/                   MOV32I R15, 0x4;                                            /* 0x180000001003dde2 */
                                                                                               /* 0x2222004200422047 */
        /*0088*/                   MOV R35, RZ;                                                /* 0x28000000fc08dde4 */
        /*0090*/                   MOV32I R14, 0x3;                                            /* 0x180000000c039de2 */
        /*0098*/                   MOV R36, RZ;                                                /* 0x28000000fc091de4 */
        /*00a0*/                   MOV32I R13, 0x2;                                            /* 0x1800000008035de2 */
        /*00a8*/                   MOV R37, RZ;                                                /* 0x28000000fc095de4 */
        /*00b0*/                   MOV32I R12, 0x1;                                            /* 0x1800000004031de2 */
        /*00b8*/                   MOV R38, RZ;                                                /* 0x28000000fc099de4 */
                                                                                               /* 0x22d042c202c042b7 */
        /*00c8*/                   IADD.X R0, R0, c[0x0][0x14c];                               /* 0x4800400530001c43 */
        /*00d0*/                   ISUB RZ.CC, R12, R16;                                       /* 0x4801000040cfdd03 */
        /*00d8*/                   SSY 0x560;                                                  /* 0x6000001200001c07 */
        /*00e0*/                   ISETP.EQ.X.AND P1, PT, R38, R0, PT;                         /* 0x190e00000263dc63 */
        /*00e8*/                   IADD32I R39.CC, R39, -0x4;                                  /* 0x0ffffffff279dc02 */
        /*00f0*/                   IADD32I.X R40, R40, -0x1;                                   /* 0x0bfffffffe8a1c42 */
        /*00f8*/                   IADD RZ.CC, -RZ, R39;                                       /* 0x480100009fffde03 */
                                                                                               /* 0x220270420042b047 */
        /*0108*/                   ISETP.NE.X.AND P0, PT, R40, RZ, PT;                         /* 0x1a8e0000fe81dc63 */
        /*0110*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0118*/                   MOV32I R2, 0x3;                                             /* 0x180000000c009de2 */
        /*0120*/                   SHL R7, R12, -0x1d;                                         /* 0x6000ffff8cc1dc03 */
        /*0128*/                   MOV32I R11, 0x0;                                            /* 0x180000000002dde2 */
        /*0130*/                   SSY 0x2f8;                                                  /* 0x6000000700001c07 */
        /*0138*/                   IADD32I RZ.CC, R2, -0x20;                                   /* 0x0fffffff802fdc02 */
                                                                                               /* 0x220042c04232b047 */
        /*0148*/                   MOV32I R17, 0x0;                                            /* 0x1800000000045de2 */
        /*0150*/                   SHL R8, R38, 0x3;                                           /* 0x6000c0000e621c03 */
        /*0158*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
        /*0160*/                   IADD32I R3.CC, R11, 0x10;                                   /* 0x0c00000040b0dc02 */
        /*0168*/                   SHR.U32 R9, R12, 0x1d;                                      /* 0x5800c00074c25c03 */
        /*0170*/                   IADD.X R6, RZ, R17;                                         /* 0x4800000047f19c43 */
        /*0178*/                   SHL R2, R16, -0x1d;                                         /* 0x6000ffff8d009c03 */
                                                                                               /* 0x2202720042921207 */
        /*0188*/                   IMAD.U32.U32 R3.CC, R12, 0x8, R3;                           /* 0x2007c00020c0dc03 */
        /*0190*/               @P1 IADD R7, R8, R9;                                            /* 0x480000002481c403 */
        /*0198*/                   SHL R8, R0, 0x3;                                            /* 0x6000c0000c021c03 */
        /*01a0*/                   SHR.U32 R9, R16, 0x1d;                                      /* 0x5800c00075025c03 */
        /*01a8*/                   IADD.X R7, R6, R7;                                          /* 0x480000001c61dc43 */
        /*01b0*/                   IMAD.U32.U32 R10.CC, R16, 0x8, R11;                         /* 0x2017c00021029c03 */
        /*01b8*/               @P1 IADD R2, R8, R9;                                            /* 0x4800000024808403 */
                                                                                               /* 0x2232328042804287 */
        /*01c8*/                   MOV R6, R3;                                                 /* 0x280000000c019de4 */
        /*01d0*/                   IADD.X R9, R2, R17;                                         /* 0x4800000044225c43 */
        /*01d8*/                   LD.E.64 R20, [R6+0x1f400];                                  /* 0x840007d000651ca5 */
        /*01e0*/                   MOV R2, R10;                                                /* 0x2800000028009de4 */
        /*01e8*/                   MOV R3, R9;                                                 /* 0x280000002400dde4 */
        /*01f0*/                   LD.E.64 R26, [R2+0x1f410];                                  /* 0x840007d040269ca5 */
        /*01f8*/                   LD.E.64 R8, [R6+0xfa00];                                    /* 0x840003e800621ca5 */
                                                                                               /* 0x2292923273f042a7 */
        /*0208*/                   LD.E.64 R28, [R2+0xfa10];                                   /* 0x840003e840271ca5 */
        /*0210*/                   MOV32I R6, 0x0;                                             /* 0x1800000000019de2 */
        /*0218*/                   MOV32I R7, 0x3fe00000;                                      /* 0x18ff80000001dde2 */
        /*0220*/                   DADD R26, -R26, R20;                                        /* 0x4800000051a69e01 */
        /*0228*/                   DADD R28, -R28, R8;                                         /* 0x4800000021c71e01 */
        /*0230*/                   DMUL R8, R26, R26;                                          /* 0x5000000069a21c01 */
        /*0238*/                   DFMA R8, R28, R28, R8;                                      /* 0x2010000071c21c01 */
                                                                                               /* 0x2292329292828047 */
        /*0248*/                   IADD32I R22, R9, -0x3500000;                                /* 0x0bf2c00000959c02 */
        /*0250*/                   MUFU.RSQ64H R11, R9;                                        /* 0xc80000001c92dc00 */
        /*0258*/                   MOV R10, R22;                                               /* 0x2800000058029de4 */
        /*0260*/                   DMUL R18, R10, R10;                                         /* 0x5000000028a49c01 */
        /*0268*/                   DFMA R20, -R8, R18, c[0x2][0x0];                            /* 0x2024880000851e01 */
        /*0270*/                   DFMA R6, R20, 0.375, R6;                                    /* 0x200ccff601419c01 */
        /*0278*/                   DMUL R20, R10, R20;                                         /* 0x5000000050a51c01 */
                                                                                               /* 0x2282028042028207 */
        /*0288*/                   DFMA R20, R6, R20, R10;                                     /* 0x2014000050651c01 */
        /*0290*/                   ISETP.LT.U32.AND P1, PT, R22, c[0x2][0x8], PT;              /* 0x188e48002163dc03 */
        /*0298*/                   DMUL R18, R8, R20;                                          /* 0x5000000050849c01 */
        /*02a0*/                   IADD32I R17, R21, -0x100000;                                /* 0x0bffc00001545c02 */
        /*02a8*/                   MOV R6, R20;                                                /* 0x2800000050019de4 */
        /*02b0*/                   DFMA R10, -R18, R18, R8;                                    /* 0x2010000049229e01 */
        /*02b8*/                   MOV R7, R17;                                                /* 0x280000004401dde4 */
                                                                                               /* 0x2232b2b2b202b207 */
        /*02c8*/                   DFMA R30, R10, R6, R18;                                     /* 0x2024000018a79c01 */
        /*02d0*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*02d8*/                   MOV R6, R20;                                                /* 0x2800000050019de4 */
        /*02e0*/                   MOV R7, R22;                                                /* 0x280000005801dde4 */
        /*02e8*/                   CAL 0x1650;                                                 /* 0x5000004d80011c07 */
        /*02f0*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*02f8*/                   LD.E.64 R8, [R2+0x10];                                      /* 0x8400000040221ca5 */
                                                                                               /* 0x2292928280420047 */
        /*0308*/                   LD.E.64 R10, [R4+0x18];                                     /* 0x8400000060429ca5 */
        /*0310*/                   DFMA R6, R30, R30, c[0x2][0x18];                            /* 0x203c880061e19c01 */
        /*0318*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
        /*0320*/                   SSY 0x3d0;                                                  /* 0x60000002a0001c07 */
        /*0328*/                   MUFU.RCP64H R19, R7;                                        /* 0xc80000001874dc00 */
        /*0330*/                   DFMA R20, -R6, R18, c[0x2][0x0];                            /* 0x2024880000651e01 */
        /*0338*/                   DFMA R20, R20, R20, R20;                                    /* 0x2028000051451c01 */
                                                                                               /* 0x22820292929293f7 */
        /*0348*/                   DFMA R18, R18, R20, R18;                                    /* 0x2024000051249c01 */
        /*0350*/                   DMUL R8, R8, c[0x2][0x10];                                  /* 0x5000480040821c01 */
        /*0358*/                   DMUL R22, R8, R10;                                          /* 0x5000000028859c01 */
        /*0360*/                   DMUL R32, R22, R18;                                         /* 0x5000000049681c01 */
        /*0368*/                   DFMA R20, -R6, R32, R22;                                    /* 0x202c000080651e01 */
        /*0370*/                   DFMA R32, R18, R20, R32;                                    /* 0x2040000051281c01 */
        /*0378*/                   FSET.GTU.AND RZ.CC, |R23|, c[0x2][0xc], PT;                 /* 0x160f4800317fdc80 */
                                                                                               /* 0x22b042b2b2b2c287 */
        /*0388*/                   FFMA R8, RZ, R7, R33;                                       /* 0x304200001ff21c00 */
        /*0390*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*0398*/               @P1 BRA CC.NEU, 0x3c0;                                          /* 0x40000000800005a7 */
        /*03a0*/                   MOV R17, R23;                                               /* 0x280000005c045de4 */
        /*03a8*/                   CAL 0x12e8;                                                 /* 0x5000003ce0011c07 */
        /*03b0*/                   MOV R32, R10;                                               /* 0x2800000028081de4 */
        /*03b8*/                   MOV R33, R11;                                               /* 0x280000002c085de4 */
                                                                                               /* 0x22929292928042b7 */
        /*03c8*/                   MUFU.RCP64H.S R9, R31;                                      /* 0xc800000019f25c10 */
        /*03d0*/                   MOV32I R8, 0x1;                                             /* 0x1800000004021de2 */
        /*03d8*/                   DMUL R22, R28, R32;                                         /* 0x5000000081c59c01 */
        /*03e0*/                   DFMA R6, -R30, R8, c[0x2][0x0];                             /* 0x2010880001e19e01 */
        /*03e8*/                   DFMA R6, R6, R6, R6;                                        /* 0x200c000018619c01 */
        /*03f0*/                   DFMA R8, R8, R6, R8;                                        /* 0x2010000018821c01 */
        /*03f8*/                   DMUL R10, R22, R8;                                          /* 0x5000000021629c01 */
                                                                                               /* 0x22b2028280420297 */
        /*0408*/                   DFMA R6, -R30, R10, R22;                                    /* 0x202c000029e19e01 */
        /*0410*/                   DFMA R8, R8, R6, R10;                                       /* 0x2014000018821c01 */
        /*0418*/                   FSET.GTU.AND RZ.CC, |R23|, c[0x2][0xc], PT;                 /* 0x160f4800317fdc80 */
        /*0420*/                   SSY 0x480;                                                  /* 0x6000000160001c07 */
        /*0428*/                   FFMA R6, RZ, R31, R9;                                       /* 0x301200007ff19c00 */
        /*0430*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
        /*0438*/                   MOV R6, R30;                                                /* 0x2800000078019de4 */
                                                                                               /* 0x22b2b042b2b2b047 */
        /*0448*/                   MOV R7, R31;                                                /* 0x280000007c01dde4 */
        /*0450*/               @P1 BRA CC.NEU, 0x478;                                          /* 0x40000000800005a7 */
        /*0458*/                   MOV R17, R23;                                               /* 0x280000005c045de4 */
        /*0460*/                   CAL 0x12e8;                                                 /* 0x5000003a00011c07 */
        /*0468*/                   MOV R8, R10;                                                /* 0x2800000028021de4 */
        /*0470*/                   MOV R9, R11;                                                /* 0x280000002c025de4 */
        /*0478*/                   NOP.S;                                                      /* 0x4000000000001df4 */
                                                                                               /* 0x2292804200423047 */
        /*0488*/                   LD.E.64 R6, [R2+0x4e210];                                   /* 0x8400138840219ca5 */
        /*0490*/                   MUFU.RCP64H R11, R31;                                       /* 0xc800000019f2dc00 */
        /*0498*/                   LD.E.64 R28, [R2+0x5dc10];                                  /* 0x8400177040271ca5 */
        /*04a0*/                   DMUL R22, R26, R32;                                         /* 0x5000000081a59c01 */
        /*04a8*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*04b0*/                   SSY 0x558;                                                  /* 0x6000000280001c07 */
        /*04b8*/                   DFMA R18, -R30, R10, c[0x2][0x0];                           /* 0x2014880001e49e01 */
                                                                                               /* 0x2203f20292929297 */
        /*04c8*/                   DFMA R20, R18, R18, R18;                                    /* 0x2024000049251c01 */
        /*04d0*/                   DFMA R24, R10, R20, R10;                                    /* 0x2014000050a61c01 */
        /*04d8*/                   DMUL R10, R24, R22;                                         /* 0x5000000059829c01 */
        /*04e0*/                   DFMA R20, -R30, R10, R22;                                   /* 0x202c000029e51e01 */
        /*04e8*/                   DFMA R10, R24, R20, R10;                                    /* 0x2014000051829c01 */
        /*04f0*/                   FSET.GTU.AND RZ.CC, |R23|, c[0x2][0xc], PT;                 /* 0x160f4800317fdc80 */
        /*04f8*/                   DADD R6, R8, R6;                                            /* 0x4800000018819c01 */
                                                                                               /* 0x22b20042b2c04287 */
        /*0508*/                   FFMA R8, RZ, R31, R11;                                      /* 0x301600007ff21c00 */
        /*0510*/                   ST.E.64 [R2+0x4e210], R6;                                   /* 0x9400138840219ca5 */
        /*0518*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*0520*/               @P1 BRA CC.NEU, 0x550;                                          /* 0x40000000a00005a7 */
        /*0528*/                   MOV R17, R23;                                               /* 0x280000005c045de4 */
        /*0530*/                   MOV R6, R30;                                                /* 0x2800000078019de4 */
        /*0538*/                   MOV R7, R31;                                                /* 0x280000007c01dde4 */
                                                                                               /* 0x22b2c2c042b2b2b7 */
        /*0548*/                   CAL 0x12e8;                                                 /* 0x5000003660011c07 */
        /*0550*/                   DADD.S R28, R28, R10;                                       /* 0x4800000029c71c11 */
        /*0558*/                   ST.E.64.S [R2+0x5dc10], R28;                                /* 0x9400177040271cb5 */
        /*0560*/                   ISUB RZ.CC, R13, R16;                                       /* 0x4801000040dfdd03 */
        /*0568*/                   SSY 0x9b8;                                                  /* 0x6000001120001c07 */
        /*0570*/                   ISETP.EQ.X.AND P1, PT, R37, R0, PT;                         /* 0x190e00000253dc63 */
        /*0578*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
                                                                                               /* 0x22b2004270420047 */
        /*0588*/                   MOV32I R2, 0x3;                                             /* 0x180000000c009de2 */
        /*0590*/                   SHL R3, R37, 0x3;                                           /* 0x6000c0000e50dc03 */
        /*0598*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*05a0*/                   SSY 0x758;                                                  /* 0x60000006c0001c07 */
        /*05a8*/                   IADD32I RZ.CC, R2, -0x20;                                   /* 0x0fffffff802fdc02 */
        /*05b0*/                   SHL R2, R13, -0x1d;                                         /* 0x6000ffff8cd09c03 */
        /*05b8*/                   MOV32I R10, 0x0;                                            /* 0x1800000000029de2 */
                                                                                               /* 0x2232220042c04237 */
        /*05c8*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
        /*05d0*/                   IADD32I R8.CC, R9, 0x10;                                    /* 0x0c00000040921c02 */
        /*05d8*/                   SHR.U32 R6, R13, 0x1d;                                      /* 0x5800c00074d19c03 */
        /*05e0*/                   IADD.X R7, RZ, R10;                                         /* 0x480000002bf1dc43 */
        /*05e8*/               @P1 IADD R2, R3, R6;                                            /* 0x4800000018308403 */
        /*05f0*/                   IMAD.U32.U32 R8.CC, R13, 0x8, R8;                           /* 0x2011c00020d21c03 */
        /*05f8*/                   SHL R11, R16, -0x1d;                                        /* 0x6000ffff8d02dc03 */
                                                                                               /* 0x2202b20200428237 */
        /*0608*/                   SHL R3, R0, 0x3;                                            /* 0x6000c0000c00dc03 */
        /*0610*/                   SHR.U32 R6, R16, 0x1d;                                      /* 0x5800c00075019c03 */
        /*0618*/                   IADD.X R7, R2, R7;                                          /* 0x480000001c21dc43 */
        /*0620*/               @P1 IADD R11, R3, R6;                                           /* 0x480000001832c403 */
        /*0628*/                   IMAD.U32.U32 R2.CC, R16, 0x8, R9;                           /* 0x2013c00021009c03 */
        /*0630*/                   MOV R6, R8;                                                 /* 0x2800000020019de4 */
        /*0638*/                   IADD.X R3, R11, R10;                                        /* 0x4800000028b0dc43 */
                                                                                               /* 0x2273f042a2323277 */
        /*0648*/                   LD.E.64 R20, [R6+0x1f400];                                  /* 0x840007d000651ca5 */
        /*0650*/                   LD.E.64 R26, [R2+0x1f410];                                  /* 0x840007d040269ca5 */
        /*0658*/                   LD.E.64 R8, [R6+0xfa00];                                    /* 0x840003e800621ca5 */
        /*0660*/                   LD.E.64 R28, [R2+0xfa10];                                   /* 0x840003e840271ca5 */
        /*0668*/                   MOV32I R6, 0x0;                                             /* 0x1800000000019de2 */
        /*0670*/                   MOV32I R7, 0x3fe00000;                                      /* 0x18ff80000001dde2 */
        /*0678*/                   DADD R26, -R26, R20;                                        /* 0x4800000051a69e01 */
                                                                                               /* 0x2292828042929237 */
        /*0688*/                   DADD R28, -R28, R8;                                         /* 0x4800000021c71e01 */
        /*0690*/                   DMUL R8, R26, R26;                                          /* 0x5000000069a21c01 */
        /*0698*/                   DFMA R8, R28, R28, R8;                                      /* 0x2010000071c21c01 */
        /*06a0*/                   IADD32I R22, R9, -0x3500000;                                /* 0x0bf2c00000959c02 */
        /*06a8*/                   MUFU.RSQ64H R11, R9;                                        /* 0xc80000001c92dc00 */
        /*06b0*/                   MOV R10, R22;                                               /* 0x2800000058029de4 */
        /*06b8*/                   DMUL R18, R10, R10;                                         /* 0x5000000028a49c01 */
                                                                                               /* 0x2202028202923297 */
        /*06c8*/                   DFMA R20, -R8, R18, c[0x2][0x0];                            /* 0x2024880000851e01 */
        /*06d0*/                   DFMA R6, R20, 0.375, R6;                                    /* 0x200ccff601419c01 */
        /*06d8*/                   DMUL R20, R10, R20;                                         /* 0x5000000050a51c01 */
        /*06e0*/                   DFMA R20, R6, R20, R10;                                     /* 0x2014000050651c01 */
        /*06e8*/                   ISETP.LT.U32.AND P1, PT, R22, c[0x2][0x8], PT;              /* 0x188e48002163dc03 */
        /*06f0*/                   DMUL R18, R8, R20;                                          /* 0x5000000050849c01 */
        /*06f8*/                   IADD32I R17, R21, -0x100000;                                /* 0x0bffc00001545c02 */
                                                                                               /* 0x22b202b202820277 */
        /*0708*/                   MOV R6, R20;                                                /* 0x2800000050019de4 */
        /*0710*/                   DFMA R10, -R18, R18, R8;                                    /* 0x2010000049229e01 */
        /*0718*/                   MOV R7, R17;                                                /* 0x280000004401dde4 */
        /*0720*/                   DFMA R30, R10, R6, R18;                                     /* 0x2024000018a79c01 */
        /*0728*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0730*/                   MOV R6, R20;                                                /* 0x2800000050019de4 */
        /*0738*/                   MOV R7, R22;                                                /* 0x280000005801dde4 */
                                                                                               /* 0x228250423042b2b7 */
        /*0748*/                   CAL 0x1650;                                                 /* 0x5000003c00011c07 */
        /*0750*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0758*/                   LD.E.64 R8, [R2+0x10];                                      /* 0x8400000040221ca5 */
        /*0760*/                   DFMA R6, R30, R30, c[0x2][0x18];                            /* 0x203c880061e19c01 */
        /*0768*/                   LD.E.64 R10, [R4+0x20];                                     /* 0x8400000080429ca5 */
        /*0770*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
        /*0778*/                   MUFU.RCP64H R19, R7;                                        /* 0xc80000001874dc00 */
                                                                                               /* 0x22929293f2029297 */
        /*0788*/                   DFMA R20, -R6, R18, c[0x2][0x0];                            /* 0x2024880000651e01 */
        /*0790*/                   DFMA R20, R20, R20, R20;                                    /* 0x2028000051451c01 */
        /*0798*/                   DFMA R18, R18, R20, R18;                                    /* 0x2024000051249c01 */
        /*07a0*/                   SSY 0x828;                                                  /* 0x6000000200001c07 */
        /*07a8*/                   DMUL R8, R8, c[0x2][0x10];                                  /* 0x5000480040821c01 */
        /*07b0*/                   DMUL R22, R8, R10;                                          /* 0x5000000028859c01 */
        /*07b8*/                   DMUL R32, R22, R18;                                         /* 0x5000000049681c01 */
                                                                                               /* 0x22b2b2c282820297 */
        /*07c8*/                   DFMA R20, -R6, R32, R22;                                    /* 0x202c000080651e01 */
        /*07d0*/                   DFMA R32, R18, R20, R32;                                    /* 0x2040000051281c01 */
        /*07d8*/                   FSET.GTU.AND RZ.CC, |R23|, c[0x2][0xc], PT;                 /* 0x160f4800317fdc80 */
        /*07e0*/                   FFMA R8, RZ, R7, R33;                                       /* 0x304200001ff21c00 */
        /*07e8*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*07f0*/               @P1 BRA CC.NEU, 0x820;                                          /* 0x40000000a00005a7 */
        /*07f8*/                   MOV R17, R23;                                               /* 0x280000005c045de4 */
                                                                                               /* 0x22928042b2b042b7 */
        /*0808*/                   CAL 0x12e8;                                                 /* 0x5000002b60011c07 */
        /*0810*/                   MOV R32, R10;                                               /* 0x2800000028081de4 */
        /*0818*/                   MOV R33, R11;                                               /* 0x280000002c085de4 */
        /*0820*/                   MUFU.RCP64H.S R9, R31;                                      /* 0xc800000019f25c10 */
        /*0828*/                   MOV32I R8, 0x1;                                             /* 0x1800000004021de2 */
        /*0830*/                   DMUL R22, R28, R32;                                         /* 0x5000000081c59c01 */
        /*0838*/                   DFMA R6, -R30, R8, c[0x2][0x0];                             /* 0x2010880001e19e01 */
                                                                                               /* 0x2280420292929297 */
        /*0848*/                   DFMA R6, R6, R6, R6;                                        /* 0x200c000018619c01 */
        /*0850*/                   DFMA R8, R8, R6, R8;                                        /* 0x2010000018821c01 */
        /*0858*/                   DMUL R10, R22, R8;                                          /* 0x5000000021629c01 */
        /*0860*/                   DFMA R6, -R30, R10, R22;                                    /* 0x202c000029e19e01 */
        /*0868*/                   DFMA R8, R8, R6, R10;                                       /* 0x2014000018821c01 */
        /*0870*/                   FSET.GTU.AND RZ.CC, |R23|, c[0x2][0xc], PT;                 /* 0x160f4800317fdc80 */
        /*0878*/                   SSY 0x8e0;                                                  /* 0x6000000180001c07 */
                                                                                               /* 0x22b2b2b042b20287 */
        /*0888*/                   FFMA R6, RZ, R31, R9;                                       /* 0x301200007ff19c00 */
        /*0890*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
        /*0898*/                   MOV R6, R30;                                                /* 0x2800000078019de4 */
        /*08a0*/                   MOV R7, R31;                                                /* 0x280000007c01dde4 */
        /*08a8*/               @P1 BRA CC.NEU, 0x8d8;                                          /* 0x40000000a00005a7 */
        /*08b0*/                   MOV R17, R23;                                               /* 0x280000005c045de4 */
        /*08b8*/                   CAL 0x12e8;                                                 /* 0x50000028a0011c07 */
                                                                                               /* 0x2200423042b2b047 */
        /*08c8*/                   MOV R8, R10;                                                /* 0x2800000028021de4 */
        /*08d0*/                   MOV R9, R11;                                                /* 0x280000002c025de4 */
        /*08d8*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*08e0*/                   LD.E.64 R6, [R2+0x4e210];                                   /* 0x8400138840219ca5 */
        /*08e8*/                   MUFU.RCP64H R11, R31;                                       /* 0xc800000019f2dc00 */
        /*08f0*/                   LD.E.64 R28, [R2+0x5dc10];                                  /* 0x8400177040271ca5 */
        /*08f8*/                   DMUL R22, R26, R32;                                         /* 0x5000000081a59c01 */
                                                                                               /* 0x2292929292928047 */
        /*0908*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*0910*/                   SSY 0x9b0;                                                  /* 0x6000000260001c07 */
        /*0918*/                   DFMA R18, -R30, R10, c[0x2][0x0];                           /* 0x2014880001e49e01 */
        /*0920*/                   DFMA R20, R18, R18, R18;                                    /* 0x2024000049251c01 */
        /*0928*/                   DFMA R24, R10, R20, R10;                                    /* 0x2014000050a61c01 */
        /*0930*/                   DMUL R10, R24, R22;                                         /* 0x5000000059829c01 */
        /*0938*/                   DFMA R20, -R30, R10, R22;                                   /* 0x202c000029e51e01 */
                                                                                               /* 0x22b2c0428203f207 */
        /*0948*/                   DFMA R10, R24, R20, R10;                                    /* 0x2014000051829c01 */
        /*0950*/                   FSET.GTU.AND RZ.CC, |R23|, c[0x2][0xc], PT;                 /* 0x160f4800317fdc80 */
        /*0958*/                   DADD R6, R8, R6;                                            /* 0x4800000018819c01 */
        /*0960*/                   FFMA R8, RZ, R31, R11;                                      /* 0x301600007ff21c00 */
        /*0968*/                   ST.E.64 [R2+0x4e210], R6;                                   /* 0x9400138840219ca5 */
        /*0970*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*0978*/               @P1 BRA CC.NEU, 0x9a8;                                          /* 0x40000000a00005a7 */
                                                                                               /* 0x22c2b2b2b2b20047 */
        /*0988*/                   MOV R17, R23;                                               /* 0x280000005c045de4 */
        /*0990*/                   MOV R6, R30;                                                /* 0x2800000078019de4 */
        /*0998*/                   MOV R7, R31;                                                /* 0x280000007c01dde4 */
        /*09a0*/                   CAL 0x12e8;                                                 /* 0x5000002500011c07 */
        /*09a8*/                   DADD.S R28, R28, R10;                                       /* 0x4800000029c71c11 */
        /*09b0*/                   ST.E.64.S [R2+0x5dc10], R28;                                /* 0x9400177040271cb5 */
        /*09b8*/                   ISUB RZ.CC, R14, R16;                                       /* 0x4801000040efdd03 */
                                                                                               /* 0x2270420042b2c047 */
        /*09c8*/                   ISETP.EQ.X.AND P1, PT, R36, R0, PT;                         /* 0x190e00000243dc63 */
        /*09d0*/                   SSY 0xe18;                                                  /* 0x6000001100001c07 */
        /*09d8*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*09e0*/                   MOV32I R2, 0x3;                                             /* 0x180000000c009de2 */
        /*09e8*/                   SHL R3, R36, 0x3;                                           /* 0x6000c0000e40dc03 */
        /*09f0*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*09f8*/                   SSY 0xbb0;                                                  /* 0x60000006c0001c07 */
                                                                                               /* 0x22c0423280423047 */
        /*0a08*/                   IADD32I RZ.CC, R2, -0x20;                                   /* 0x0fffffff802fdc02 */
        /*0a10*/                   SHL R2, R14, -0x1d;                                         /* 0x6000ffff8ce09c03 */
        /*0a18*/                   MOV32I R10, 0x0;                                            /* 0x1800000000029de2 */
        /*0a20*/                   SHR.U32 R6, R14, 0x1d;                                      /* 0x5800c00074e19c03 */
        /*0a28*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
        /*0a30*/                   IADD32I R8.CC, R9, 0x10;                                    /* 0x0c00000040921c02 */
        /*0a38*/                   SHL R11, R16, -0x1d;                                        /* 0x6000ffff8d02dc03 */
                                                                                               /* 0x2200428232220047 */
        /*0a48*/                   IADD.X R7, RZ, R10;                                         /* 0x480000002bf1dc43 */
        /*0a50*/               @P1 IADD R2, R3, R6;                                            /* 0x4800000018308403 */
        /*0a58*/                   IMAD.U32.U32 R8.CC, R14, 0x8, R8;                           /* 0x2011c00020e21c03 */
        /*0a60*/                   SHL R3, R0, 0x3;                                            /* 0x6000c0000c00dc03 */
        /*0a68*/                   SHR.U32 R6, R16, 0x1d;                                      /* 0x5800c00075019c03 */
        /*0a70*/                   IADD.X R7, R2, R7;                                          /* 0x480000001c21dc43 */
        /*0a78*/               @P1 IADD R11, R3, R6;                                           /* 0x480000001832c403 */
                                                                                               /* 0x22a232328042b207 */
        /*0a88*/                   IMAD.U32.U32 R2.CC, R16, 0x8, R9;                           /* 0x2013c00021009c03 */
        /*0a90*/                   MOV R6, R8;                                                 /* 0x2800000020019de4 */
        /*0a98*/                   IADD.X R3, R11, R10;                                        /* 0x4800000028b0dc43 */
        /*0aa0*/                   LD.E.64 R20, [R6+0x1f400];                                  /* 0x840007d000651ca5 */
        /*0aa8*/                   LD.E.64 R26, [R2+0x1f410];                                  /* 0x840007d040269ca5 */
        /*0ab0*/                   LD.E.64 R8, [R6+0xfa00];                                    /* 0x840003e800621ca5 */
        /*0ab8*/                   LD.E.64 R28, [R2+0xfa10];                                   /* 0x840003e840271ca5 */
                                                                                               /* 0x228292923273f047 */
        /*0ac8*/                   MOV32I R6, 0x0;                                             /* 0x1800000000019de2 */
        /*0ad0*/                   MOV32I R7, 0x3fe00000;                                      /* 0x18ff80000001dde2 */
        /*0ad8*/                   DADD R26, -R26, R20;                                        /* 0x4800000051a69e01 */
        /*0ae0*/                   DADD R28, -R28, R8;                                         /* 0x4800000021c71e01 */
        /*0ae8*/                   DMUL R8, R26, R26;                                          /* 0x5000000069a21c01 */
        /*0af0*/                   DFMA R8, R28, R28, R8;                                      /* 0x2010000071c21c01 */
        /*0af8*/                   IADD32I R22, R9, -0x3500000;                                /* 0x0bf2c00000959c02 */
                                                                                               /* 0x2202923292928047 */
        /*0b08*/                   MUFU.RSQ64H R11, R9;                                        /* 0xc80000001c92dc00 */
        /*0b10*/                   MOV R10, R22;                                               /* 0x2800000058029de4 */
        /*0b18*/                   DMUL R18, R10, R10;                                         /* 0x5000000028a49c01 */
        /*0b20*/                   DFMA R20, -R8, R18, c[0x2][0x0];                            /* 0x2024880000851e01 */
        /*0b28*/                   DFMA R6, R20, 0.375, R6;                                    /* 0x200ccff601419c01 */
        /*0b30*/                   DMUL R20, R10, R20;                                         /* 0x5000000050a51c01 */
        /*0b38*/                   DFMA R20, R6, R20, R10;                                     /* 0x2014000050651c01 */
                                                                                               /* 0x2202820280420287 */
        /*0b48*/                   ISETP.LT.U32.AND P1, PT, R22, c[0x2][0x8], PT;              /* 0x188e48002163dc03 */
        /*0b50*/                   DMUL R18, R8, R20;                                          /* 0x5000000050849c01 */
        /*0b58*/                   IADD32I R17, R21, -0x100000;                                /* 0x0bffc00001545c02 */
        /*0b60*/                   MOV R6, R20;                                                /* 0x2800000050019de4 */
        /*0b68*/                   DFMA R10, -R18, R18, R8;                                    /* 0x2010000049229e01 */
        /*0b70*/                   MOV R7, R17;                                                /* 0x280000004401dde4 */
        /*0b78*/                   DFMA R30, R10, R6, R18;                                     /* 0x2024000018a79c01 */
                                                                                               /* 0x223042b2b2b202b7 */
        /*0b88*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0b90*/                   MOV R6, R20;                                                /* 0x2800000050019de4 */
        /*0b98*/                   MOV R7, R22;                                                /* 0x280000005801dde4 */
        /*0ba0*/                   CAL 0x1650;                                                 /* 0x5000002aa0011c07 */
        /*0ba8*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0bb0*/                   LD.E.64 R8, [R2+0x10];                                      /* 0x8400000040221ca5 */
        /*0bb8*/                   DFMA R6, R30, R30, c[0x2][0x18];                            /* 0x203c880061e19c01 */
                                                                                               /* 0x23f2929280425047 */
        /*0bc8*/                   LD.E.64 R10, [R4+0x28];                                     /* 0x84000000a0429ca5 */
        /*0bd0*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
        /*0bd8*/                   MUFU.RCP64H R19, R7;                                        /* 0xc80000001874dc00 */
        /*0be0*/                   SSY 0xc80;                                                  /* 0x6000000260001c07 */
        /*0be8*/                   DFMA R20, -R6, R18, c[0x2][0x0];                            /* 0x2024880000651e01 */
        /*0bf0*/                   DFMA R20, R20, R20, R20;                                    /* 0x2028000051451c01 */
        /*0bf8*/                   DFMA R18, R18, R20, R18;                                    /* 0x2024000051249c01 */
                                                                                               /* 0x2282820292929297 */
        /*0c08*/                   DMUL R8, R8, c[0x2][0x10];                                  /* 0x5000480040821c01 */
        /*0c10*/                   DMUL R22, R8, R10;                                          /* 0x5000000028859c01 */
        /*0c18*/                   DMUL R32, R22, R18;                                         /* 0x5000000049681c01 */
        /*0c20*/                   DFMA R20, -R6, R32, R22;                                    /* 0x202c000080651e01 */
        /*0c28*/                   DFMA R32, R18, R20, R32;                                    /* 0x2040000051281c01 */
        /*0c30*/                   FSET.GTU.AND RZ.CC, |R23|, c[0x2][0xc], PT;                 /* 0x160f4800317fdc80 */
        /*0c38*/                   FFMA R8, RZ, R7, R33;                                       /* 0x304200001ff21c00 */
                                                                                               /* 0x22b2b042b2b2b2c7 */
        /*0c48*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*0c50*/               @P1 BRA CC.NEU, 0xc78;                                          /* 0x40000000800005a7 */
        /*0c58*/                   MOV R17, R23;                                               /* 0x280000005c045de4 */
        /*0c60*/                   CAL 0x12e8;                                                 /* 0x5000001a00011c07 */
        /*0c68*/                   MOV R32, R10;                                               /* 0x2800000028081de4 */
        /*0c70*/                   MOV R33, R11;                                               /* 0x280000002c085de4 */
        /*0c78*/                   MUFU.RCP64H.S R9, R31;                                      /* 0xc800000019f25c10 */
                                                                                               /* 0x2292929292928047 */
        /*0c88*/                   MOV32I R8, 0x1;                                             /* 0x1800000004021de2 */
        /*0c90*/                   DMUL R22, R28, R32;                                         /* 0x5000000081c59c01 */
        /*0c98*/                   DFMA R6, -R30, R8, c[0x2][0x0];                             /* 0x2010880001e19e01 */
        /*0ca0*/                   DFMA R6, R6, R6, R6;                                        /* 0x200c000018619c01 */
        /*0ca8*/                   DFMA R8, R8, R6, R8;                                        /* 0x2010000018821c01 */
        /*0cb0*/                   DMUL R10, R22, R8;                                          /* 0x5000000021629c01 */
        /*0cb8*/                   DFMA R6, -R30, R10, R22;                                    /* 0x202c000029e19e01 */
                                                                                               /* 0x22a2020282804207 */
        /*0cc8*/                   DFMA R8, R8, R6, R10;                                       /* 0x2014000018821c01 */
        /*0cd0*/                   FSET.GTU.AND RZ.CC, |R23|, c[0x2][0xc], PT;                 /* 0x160f4800317fdc80 */
        /*0cd8*/                   SSY 0xd38;                                                  /* 0x6000000160001c07 */
        /*0ce0*/                   FFMA R6, RZ, R31, R9;                                       /* 0x301200007ff19c00 */
        /*0ce8*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
        /*0cf0*/                   MOV R6, R30;                                                /* 0x2800000078019de4 */
        /*0cf8*/                   MOV R7, R31;                                                /* 0x280000007c01dde4 */
                                                                                               /* 0x2232b2b042b2b2b7 */
        /*0d08*/               @P1 BRA CC.NEU, 0xd30;                                          /* 0x40000000800005a7 */
        /*0d10*/                   MOV R17, R23;                                               /* 0x280000005c045de4 */
        /*0d18*/                   CAL 0x12e8;                                                 /* 0x5000001720011c07 */
        /*0d20*/                   MOV R8, R10;                                                /* 0x2800000028021de4 */
        /*0d28*/                   MOV R9, R11;                                                /* 0x280000002c025de4 */
        /*0d30*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0d38*/                   LD.E.64 R6, [R2+0x4e210];                                   /* 0x8400138840219ca5 */
                                                                                               /* 0x2292929280420047 */
        /*0d48*/                   LD.E.64 R28, [R2+0x5dc10];                                  /* 0x8400177040271ca5 */
        /*0d50*/                   MUFU.RCP64H R11, R31;                                       /* 0xc800000019f2dc00 */
        /*0d58*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*0d60*/                   DMUL R22, R26, R32;                                         /* 0x5000000081a59c01 */
        /*0d68*/                   DFMA R18, -R30, R10, c[0x2][0x0];                           /* 0x2014880001e49e01 */
        /*0d70*/                   DFMA R20, R18, R18, R18;                                    /* 0x2024000049251c01 */
        /*0d78*/                   DFMA R24, R10, R20, R10;                                    /* 0x2014000050a61c01 */
                                                                                               /* 0x228203f042029297 */
        /*0d88*/                   DMUL R10, R24, R22;                                         /* 0x5000000059829c01 */
        /*0d90*/                   DFMA R20, -R30, R10, R22;                                   /* 0x202c000029e51e01 */
        /*0d98*/                   DFMA R10, R24, R20, R10;                                    /* 0x2014000051829c01 */
        /*0da0*/                   FSET.GTU.AND RZ.CC, |R23|, c[0x2][0xc], PT;                 /* 0x160f4800317fdc80 */
        /*0da8*/                   SSY 0xe10;                                                  /* 0x6000000180001c07 */
        /*0db0*/                   DADD R6, R8, R6;                                            /* 0x4800000018819c01 */
        /*0db8*/                   FFMA R8, RZ, R31, R11;                                      /* 0x301600007ff21c00 */
                                                                                               /* 0x22b2b20042b2c047 */
        /*0dc8*/                   ST.E.64 [R2+0x4e210], R6;                                   /* 0x9400138840219ca5 */
        /*0dd0*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*0dd8*/               @P1 BRA CC.NEU, 0xe00;                                          /* 0x40000000800005a7 */
        /*0de0*/                   MOV R17, R23;                                               /* 0x280000005c045de4 */
        /*0de8*/                   MOV R6, R30;                                                /* 0x2800000078019de4 */
        /*0df0*/                   MOV R7, R31;                                                /* 0x280000007c01dde4 */
        /*0df8*/                   CAL 0x12e8;                                                 /* 0x50000013a0011c07 */
                                                                                               /* 0x2202b2c2c042b2b7 */
        /*0e08*/                   DADD.S R28, R28, R10;                                       /* 0x4800000029c71c11 */
        /*0e10*/                   ST.E.64.S [R2+0x5dc10], R28;                                /* 0x9400177040271cb5 */
        /*0e18*/                   ISUB RZ.CC, R15, R16;                                       /* 0x4801000040ffdd03 */
        /*0e20*/                   SSY 0x1278;                                                 /* 0x6000001140001c07 */
        /*0e28*/                   ISETP.EQ.X.AND P1, PT, R35, R0, PT;                         /* 0x190e00000233dc63 */
        /*0e30*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0e38*/                   MOV32I R2, 0x3;                                             /* 0x180000000c009de2 */
                                                                                               /* 0x2232b04200427047 */
        /*0e48*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*0e50*/                   SHL R3, R35, 0x3;                                           /* 0x6000c0000e30dc03 */
        /*0e58*/                   IADD32I RZ.CC, R2, -0x20;                                   /* 0x0fffffff802fdc02 */
        /*0e60*/                   SHL R2, R15, -0x1d;                                         /* 0x6000ffff8cf09c03 */
        /*0e68*/                   MOV32I R10, 0x0;                                            /* 0x1800000000029de2 */
        /*0e70*/                   SSY 0x1018;                                                 /* 0x6000000680001c07 */
        /*0e78*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
                                                                                               /* 0x223232220042c047 */
        /*0e88*/                   IADD32I R8.CC, R9, 0x10;                                    /* 0x0c00000040921c02 */
        /*0e90*/                   SHR.U32 R6, R15, 0x1d;                                      /* 0x5800c00074f19c03 */
        /*0e98*/                   IADD.X R7, RZ, R10;                                         /* 0x480000002bf1dc43 */
        /*0ea0*/               @P1 IADD R2, R3, R6;                                            /* 0x4800000018308403 */
        /*0ea8*/                   IMAD.U32.U32 R8.CC, R15, 0x8, R8;                           /* 0x2011c00020f21c03 */
        /*0eb0*/                   SHL R11, R16, -0x1d;                                        /* 0x6000ffff8d02dc03 */
        /*0eb8*/                   SHL R3, R0, 0x3;                                            /* 0x6000c0000c00dc03 */
                                                                                               /* 0x228042b202004287 */
        /*0ec8*/                   SHR.U32 R6, R16, 0x1d;                                      /* 0x5800c00075019c03 */
        /*0ed0*/                   IADD.X R7, R2, R7;                                          /* 0x480000001c21dc43 */
        /*0ed8*/               @P1 IADD R11, R3, R6;                                           /* 0x480000001832c403 */
        /*0ee0*/                   IMAD.U32.U32 R2.CC, R16, 0x8, R9;                           /* 0x2013c00021009c03 */
        /*0ee8*/                   MOV R6, R8;                                                 /* 0x2800000020019de4 */
        /*0ef0*/                   IADD.X R3, R11, R10;                                        /* 0x4800000028b0dc43 */
        /*0ef8*/                   LD.E.64 R8, [R6+0x1f400];                                   /* 0x840007d000621ca5 */
                                                                                               /* 0x223273f042a23237 */
        /*0f08*/                   LD.E.64 R26, [R2+0x1f410];                                  /* 0x840007d040269ca5 */
        /*0f10*/                   LD.E.64 R10, [R6+0xfa00];                                   /* 0x840003e800629ca5 */
        /*0f18*/                   LD.E.64 R32, [R2+0xfa10];                                   /* 0x840003e840281ca5 */
        /*0f20*/                   MOV32I R6, 0x0;                                             /* 0x1800000000019de2 */
        /*0f28*/                   MOV32I R7, 0x3fe00000;                                      /* 0x18ff80000001dde2 */
        /*0f30*/                   DADD R26, -R26, R8;                                         /* 0x4800000021a69e01 */
        /*0f38*/                   DADD R32, -R32, R10;                                        /* 0x480000002a081e01 */
                                                                                               /* 0x2292928280429297 */
        /*0f48*/                   DMUL R10, R26, R26;                                         /* 0x5000000069a29c01 */
        /*0f50*/                   DFMA R8, R32, R32, R10;                                     /* 0x2014000082021c01 */
        /*0f58*/                   IADD32I R22, R9, -0x3500000;                                /* 0x0bf2c00000959c02 */
        /*0f60*/                   MUFU.RSQ64H R11, R9;                                        /* 0xc80000001c92dc00 */
        /*0f68*/                   MOV R10, R22;                                               /* 0x2800000058029de4 */
        /*0f70*/                   DMUL R18, R10, R10;                                         /* 0x5000000028a49c01 */
        /*0f78*/                   DFMA R20, -R8, R18, c[0x2][0x0];                            /* 0x2024880000851e01 */
                                                                                               /* 0x2280420282029237 */
        /*0f88*/                   DFMA R6, R20, 0.375, R6;                                    /* 0x200ccff601419c01 */
        /*0f90*/                   DMUL R20, R10, R20;                                         /* 0x5000000050a51c01 */
        /*0f98*/                   DFMA R20, R6, R20, R10;                                     /* 0x2014000050651c01 */
        /*0fa0*/                   ISETP.LT.U32.AND P1, PT, R22, c[0x2][0x8], PT;              /* 0x188e48002163dc03 */
        /*0fa8*/                   DMUL R18, R8, R20;                                          /* 0x5000000050849c01 */
        /*0fb0*/                   IADD32I R17, R21, -0x100000;                                /* 0x0bffc00001545c02 */
        /*0fb8*/                   MOV R6, R20;                                                /* 0x2800000050019de4 */
                                                                                               /* 0x22b2b202b2028207 */
        /*0fc8*/                   DFMA R10, -R18, R18, R8;                                    /* 0x2010000049229e01 */
        /*0fd0*/                   MOV R7, R17;                                                /* 0x280000004401dde4 */
        /*0fd8*/                   DFMA R28, R10, R6, R18;                                     /* 0x2024000018a71c01 */
        /*0fe0*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0fe8*/                   MOV R6, R20;                                                /* 0x2800000050019de4 */
        /*0ff0*/                   MOV R7, R22;                                                /* 0x280000005801dde4 */
        /*0ff8*/                   CAL 0x1650;                                                 /* 0x5000001940011c07 */
                                                                                               /* 0x228250423042b047 */
        /*1008*/                   MOV R28, R30;                                               /* 0x2800000078071de4 */
        /*1010*/                   MOV.S R29, R31;                                             /* 0x280000007c075df4 */
        /*1018*/                   LD.E.64 R8, [R2+0x10];                                      /* 0x8400000040221ca5 */
        /*1020*/                   DFMA R6, R28, R28, c[0x2][0x18];                            /* 0x2038880061c19c01 */
        /*1028*/                   LD.E.64 R10, [R4+0x30];                                     /* 0x84000000c0429ca5 */
        /*1030*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
        /*1038*/                   MUFU.RCP64H R19, R7;                                        /* 0xc80000001874dc00 */
                                                                                               /* 0x22929293f2029297 */
        /*1048*/                   DFMA R20, -R6, R18, c[0x2][0x0];                            /* 0x2024880000651e01 */
        /*1050*/                   DFMA R20, R20, R20, R20;                                    /* 0x2028000051451c01 */
        /*1058*/                   DFMA R18, R18, R20, R18;                                    /* 0x2024000051249c01 */
        /*1060*/                   SSY 0x10e8;                                                 /* 0x6000000200001c07 */
        /*1068*/                   DMUL R8, R8, c[0x2][0x10];                                  /* 0x5000480040821c01 */
        /*1070*/                   DMUL R22, R8, R10;                                          /* 0x5000000028859c01 */
        /*1078*/                   DMUL R8, R22, R18;                                          /* 0x5000000049621c01 */
                                                                                               /* 0x22b2b2c282820297 */
        /*1088*/                   DFMA R30, -R6, R8, R22;                                     /* 0x202c000020679e01 */
        /*1090*/                   DFMA R30, R18, R30, R8;                                     /* 0x2010000079279c01 */
        /*1098*/                   FSET.GTU.AND RZ.CC, |R23|, c[0x2][0xc], PT;                 /* 0x160f4800317fdc80 */
        /*10a0*/                   FFMA R8, RZ, R7, R31;                                       /* 0x303e00001ff21c00 */
        /*10a8*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*10b0*/               @P1 BRA CC.NEU, 0x10e0;                                         /* 0x40000000a00005a7 */
        /*10b8*/                   MOV R17, R23;                                               /* 0x280000005c045de4 */
                                                                                               /* 0x22928042b2b042b7 */
        /*10c8*/                   CAL 0x12e8;                                                 /* 0x5000000860011c07 */
        /*10d0*/                   MOV R30, R10;                                               /* 0x2800000028079de4 */
        /*10d8*/                   MOV R31, R11;                                               /* 0x280000002c07dde4 */
        /*10e0*/                   MUFU.RCP64H.S R7, R29;                                      /* 0xc800000019d1dc10 */
        /*10e8*/                   MOV32I R6, 0x1;                                             /* 0x1800000004019de2 */
        /*10f0*/                   DMUL R22, R32, R30;                                         /* 0x500000007a059c01 */
        /*10f8*/                   DFMA R8, -R28, R6, c[0x2][0x0];                             /* 0x200c880001c21e01 */
                                                                                               /* 0x2280420292929297 */
        /*1108*/                   DFMA R8, R8, R8, R8;                                        /* 0x2010000020821c01 */
        /*1110*/                   DFMA R8, R6, R8, R6;                                        /* 0x200c000020621c01 */
        /*1118*/                   DMUL R6, R22, R8;                                           /* 0x5000000021619c01 */
        /*1120*/                   DFMA R10, -R28, R6, R22;                                    /* 0x202c000019c29e01 */
        /*1128*/                   DFMA R8, R8, R10, R6;                                       /* 0x200c000028821c01 */
        /*1130*/                   FSET.GTU.AND RZ.CC, |R23|, c[0x2][0xc], PT;                 /* 0x160f4800317fdc80 */
        /*1138*/                   SSY 0x11a0;                                                 /* 0x6000000180001c07 */
                                                                                               /* 0x22b2b2b042b20287 */
        /*1148*/                   FFMA R6, RZ, R29, R9;                                       /* 0x3012000077f19c00 */
        /*1150*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
        /*1158*/                   MOV R6, R28;                                                /* 0x2800000070019de4 */
        /*1160*/                   MOV R7, R29;                                                /* 0x280000007401dde4 */
        /*1168*/               @P1 BRA CC.NEU, 0x1198;                                         /* 0x40000000a00005a7 */
        /*1170*/                   MOV R17, R23;                                               /* 0x280000005c045de4 */
        /*1178*/                   CAL 0x12e8;                                                 /* 0x50000005a0011c07 */
                                                                                               /* 0x2200423042b2b047 */
        /*1188*/                   MOV R8, R10;                                                /* 0x2800000028021de4 */
        /*1190*/                   MOV R9, R11;                                                /* 0x280000002c025de4 */
        /*1198*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*11a0*/                   LD.E.64 R6, [R2+0x4e210];                                   /* 0x8400138840219ca5 */
        /*11a8*/                   MUFU.RCP64H R11, R29;                                       /* 0xc800000019d2dc00 */
        /*11b0*/                   LD.E.64 R32, [R2+0x5dc10];                                  /* 0x8400177040281ca5 */
        /*11b8*/                   DMUL R22, R26, R30;                                         /* 0x5000000079a59c01 */
                                                                                               /* 0x2292929292928047 */
        /*11c8*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*11d0*/                   SSY 0x1270;                                                 /* 0x6000000260001c07 */
        /*11d8*/                   DFMA R18, -R28, R10, c[0x2][0x0];                           /* 0x2014880001c49e01 */
        /*11e0*/                   DFMA R20, R18, R18, R18;                                    /* 0x2024000049251c01 */
        /*11e8*/                   DFMA R24, R10, R20, R10;                                    /* 0x2014000050a61c01 */
        /*11f0*/                   DMUL R10, R24, R22;                                         /* 0x5000000059829c01 */
        /*11f8*/                   DFMA R20, -R28, R10, R22;                                   /* 0x202c000029c51e01 */
                                                                                               /* 0x22b2c0428203f207 */
        /*1208*/                   DFMA R10, R24, R20, R10;                                    /* 0x2014000051829c01 */
        /*1210*/                   FSET.GTU.AND RZ.CC, |R23|, c[0x2][0xc], PT;                 /* 0x160f4800317fdc80 */
        /*1218*/                   DADD R6, R8, R6;                                            /* 0x4800000018819c01 */
        /*1220*/                   FFMA R8, RZ, R29, R11;                                      /* 0x3016000077f21c00 */
        /*1228*/                   ST.E.64 [R2+0x4e210], R6;                                   /* 0x9400138840219ca5 */
        /*1230*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*1238*/               @P1 BRA CC.NEU, 0x1268;                                         /* 0x40000000a00005a7 */
                                                                                               /* 0x22c2b2b2b2b20047 */
        /*1248*/                   MOV R17, R23;                                               /* 0x280000005c045de4 */
        /*1250*/                   MOV R6, R28;                                                /* 0x2800000070019de4 */
        /*1258*/                   MOV R7, R29;                                                /* 0x280000007401dde4 */
        /*1260*/                   CAL 0x12e8;                                                 /* 0x5000000200011c07 */
        /*1268*/                   DADD.S R32, R32, R10;                                       /* 0x480000002a081c11 */
        /*1270*/                   ST.E.64.S [R2+0x5dc10], R32;                                /* 0x9400177040281cb5 */
        /*1278*/                   IADD32I R4.CC, R4, 0x20;                                    /* 0x0c00000080411c02 */
                                                                                               /* 0x2202c042c042c207 */
        /*1288*/                   IADD.X R5, RZ, R5;                                          /* 0x4800000017f15c43 */
        /*1290*/                   IADD32I R15.CC, R15, 0x4;                                   /* 0x0c00000010f3dc02 */
        /*1298*/                   IADD.X R35, RZ, R35;                                        /* 0x480000008ff8dc43 */
        /*12a0*/                   IADD32I R14.CC, R14, 0x4;                                   /* 0x0c00000010e39c02 */
        /*12a8*/                   IADD.X R36, RZ, R36;                                        /* 0x4800000093f91c43 */
        /*12b0*/                   IADD32I R13.CC, R13, 0x4;                                   /* 0x0c00000010d35c02 */
        /*12b8*/                   IADD.X R37, RZ, R37;                                        /* 0x4800000097f95c43 */
                                                                                               /* 0x22720042e2e042c7 */
        /*12c8*/                   IADD32I R12.CC, R12, 0x4;                                   /* 0x0c00000010c31c02 */
        /*12d0*/                   IADD.X R38, RZ, R38;                                        /* 0x480000009bf99c43 */
        /*12d8*/               @P0 BRA 0xd0;                                                   /* 0x4003ffb7c00001e7 */
        /*12e0*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*12e8*/                   LOP32I.AND R8, R7, 0x40000000;                              /* 0x3900000000721c02 */
        /*12f0*/                   LOP32I.AND R34, R17, 0x7f800000;                            /* 0x39fe000001189c02 */
        /*12f8*/                   MOV32I R41, 0x1ff00000;                                     /* 0x187fc000000a5de2 */
                                                                                               /* 0x228042928042c047 */
        /*1308*/                   ISETP.LT.U32.AND P1, PT, R8, c[0x2][0x24], PT;              /* 0x188e48009083dc03 */
        /*1310*/                   MOV R8, RZ;                                                 /* 0x28000000fc021de4 */
        /*1318*/                   SEL R9, R41, c[0x2][0x28], !P1;                             /* 0x20124800a2925c04 */
        /*1320*/                   PBK 0x1630;                                                 /* 0x6800000c20001c07 */
        /*1328*/                   DMUL R20, R6, R8;                                           /* 0x5000000020651c01 */
        /*1330*/                   MUFU.RCP64H R18, R21;                                       /* 0xc800000019549c00 */
        /*1338*/                   ISETP.LT.U32.AND P1, PT, R34, c[0x2][0x2c], PT;             /* 0x188e4800b223dc03 */
                                                                                               /* 0x2280420282028047 */
        /*1348*/                   MOV R19, R18;                                               /* 0x280000004804dde4 */
        /*1350*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
        /*1358*/                   DFMA R10, -R20, R18, c[0x2][0x0];                           /* 0x2024880001429e01 */
        /*1360*/                   SEL R24, R41, c[0x2][0x28], !P1;                            /* 0x20124800a2961c04 */
        /*1368*/                   DFMA R10, R10, R10, R10;                                    /* 0x2014000028a29c01 */
        /*1370*/                   MOV R25, R24;                                               /* 0x2800000060065de4 */
        /*1378*/                   MOV R24, RZ;                                                /* 0x28000000fc061de4 */
                                                                                               /* 0x2292929292804207 */
        /*1388*/                   DFMA R18, R18, R10, R18;                                    /* 0x2024000029249c01 */
        /*1390*/                   MOV R10, R22;                                               /* 0x2800000058029de4 */
        /*1398*/                   MOV R11, R17;                                               /* 0x280000004402dde4 */
        /*13a0*/                   DMUL R24, R10, R24;                                         /* 0x5000000060a61c01 */
        /*13a8*/                   DMUL R22, R24, R18;                                         /* 0x5000000049859c01 */
        /*13b0*/                   DFMA R20, -R20, R22, R24;                                   /* 0x2030000059451e01 */
        /*13b8*/                   DFMA R22, R18, R20, R22;                                    /* 0x202c000051259c01 */
                                                                                               /* 0x2242304282c2b2d7 */
        /*13c8*/                   DSETP.LEU.AND P1, PT, |R22|, RZ, PT;                        /* 0x1d8e0000fd63dc81 */
        /*13d0*/               @P1 BRA 0x15b0;                                                 /* 0x40000007600005e7 */
        /*13d8*/                   ISETP.GT.U32.AND P1, PT, R34, c[0x2][0x30], PT;             /* 0x1a0e4800c223dc03 */
        /*13e0*/                   SEL R18, R41, c[0x2][0x28], !P1;                            /* 0x20124800a2949c04 */
        /*13e8*/                   MOV R19, R18;                                               /* 0x280000004804dde4 */
        /*13f0*/                   MOV R18, RZ;                                                /* 0x28000000fc049de4 */
        /*13f8*/                   DMUL R20, R8, R22;                                          /* 0x5000000058851c01 */
                                                                                               /* 0x2202d29242424247 */
        /*1408*/                   DMUL R22, R22, R18;                                         /* 0x5000000049659c01 */
        /*1410*/                   DMUL R18, R18, R20;                                         /* 0x5000000051249c01 */
        /*1418*/                   DMUL R22, R8, R22;                                          /* 0x5000000058859c01 */
        /*1420*/                   DFMA R8, R6, R18, -R10;                                     /* 0x2014000048621d01 */
        /*1428*/                   DFMA R20, R6, R22, -R10;                                    /* 0x2014000058651d01 */
        /*1430*/                   DSETP.GT.AND P1, PT, |R8|, |R20|, PT;                       /* 0x1a0e00005083dcc1 */
        /*1438*/                   SEL R8, R23, R19, P1;                                       /* 0x200200004d721c04 */
                                                                                               /* 0x22020042b042c277 */
        /*1448*/                   SEL R20, R22, R18, P1;                                      /* 0x2002000049651c04 */
        /*1450*/                   FSETP.GTU.AND P1, PT, |R8|, 1.469367938527859385e-39, PT;   /* 0x260ec0040083dc80 */
        /*1458*/                   MOV R21, R8;                                                /* 0x2800000020055de4 */
        /*1460*/               @P1 BRK;                                                        /* 0xa8000000000005e7 */
        /*1468*/                   FSETP.LT.AND P1, PT, |R11|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000b3dc80 */
        /*1470*/                   LOP32I.AND R17, R20, 0xfffffffe;                            /* 0x3bfffffff9445c02 */
        /*1478*/                   MOV32I R18, 0x3ff00000;                                     /* 0x18ffc00000049de2 */
                                                                                               /* 0x223280428042a047 */
        /*1488*/                   LOP32I.OR R9, R20, 0x1;                                     /* 0x3800000005425c42 */
        /*1490*/                   MOV R21, R8;                                                /* 0x2800000020055de4 */
        /*1498*/                   SEL R18, R18, c[0x2][0x34], !P1;                            /* 0x20124800d1249c04 */
        /*14a0*/                   MOV R20, R17;                                               /* 0x2800000044051de4 */
        /*14a8*/                   MOV R19, R18;                                               /* 0x280000004804dde4 */
        /*14b0*/                   MOV R18, RZ;                                                /* 0x28000000fc049de4 */
        /*14b8*/                   DMUL R10, R10, R18;                                         /* 0x5000000048a29c01 */
                                                                                               /* 0x2282d29232804207 */
        /*14c8*/                   DMUL R6, R6, R18;                                           /* 0x5000000048619c01 */
        /*14d0*/                   MOV R18, R9;                                                /* 0x2800000024049de4 */
        /*14d8*/                   MOV R19, R8;                                                /* 0x280000002004dde4 */
        /*14e0*/                   DFMA R18, R18, R6, -R10;                                    /* 0x2014000019249d01 */
        /*14e8*/                   DFMA R20, R20, R6, -R10;                                    /* 0x2014000019451d01 */
        /*14f0*/                   DSETP.GT.AND P1, PT, |R20|, |R18|, PT;                      /* 0x1a0e00004943dcc1 */
        /*14f8*/                   SEL R20, R9, R17, P1;                                       /* 0x2002000044951c04 */
                                                                                               /* 0x220042c20042c047 */
        /*1508*/                   IADD32I R21.CC, R20, 0x1;                                   /* 0x0c00000005455c02 */
        /*1510*/                   LOP32I.AND R9, R20, 0x1;                                    /* 0x3800000005425c02 */
        /*1518*/                   ISETP.EQ.U32.AND P1, PT, R9, 0x1, PT;                       /* 0x190ec0000493dc03 */
        /*1520*/                   IADD.X R17, RZ, R8;                                         /* 0x4800000023f45c43 */
        /*1528*/                   IADD32I R18.CC, R20, -0x1;                                  /* 0x0ffffffffd449c02 */
        /*1530*/                   SEL R21, R20, R21, !P1;                                     /* 0x2012000055455c04 */
        /*1538*/                   IADD32I.X R9, R8, -0x1;                                     /* 0x0bfffffffc825c42 */
                                                                                               /* 0x2232720202026207 */
        /*1548*/                   SEL R17, R8, R17, !P1;                                      /* 0x2012000044845c04 */
        /*1550*/                   SEL R20, R18, R20, !P1;                                     /* 0x2012000051251c04 */
        /*1558*/                   SEL R9, R9, R8, !P1;                                        /* 0x2012000020925c04 */
        /*1560*/                   MOV R18, R21;                                               /* 0x2800000054049de4 */
        /*1568*/                   MOV R19, R17;                                               /* 0x280000004404dde4 */
        /*1570*/                   MOV R8, R20;                                                /* 0x2800000050021de4 */
        /*1578*/                   DFMA R18, R6, R18, -R10;                                    /* 0x2014000048649d01 */
                                                                                               /* 0x22b2d2e04202d297 */
        /*1588*/                   DFMA R6, R6, R8, -R10;                                      /* 0x2014000020619d01 */
        /*1590*/                   DSETP.GT.AND P1, PT, |R18|, |R6|, PT;                       /* 0x1a0e00001923dcc1 */
        /*1598*/                   SEL R20, R20, R21, P1;                                      /* 0x2002000055451c04 */
        /*15a0*/                   SEL R21, R9, R17, P1;                                       /* 0x2002000044955c04 */
        /*15a8*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*15b0*/                   DSETP.EQ.AND P1, PT, R22, RZ, PT;                           /* 0x190e0000fd63dc01 */
        /*15b8*/               @P1 BRA 0x1620;                                                 /* 0x40000001800005e7 */
                                                                                               /* 0x22027202d2d28047 */
        /*15c8*/                   MUFU.RCP64H R9, R7;                                         /* 0xc800000018725c00 */
        /*15d0*/                   MOV R8, RZ;                                                 /* 0x28000000fc021de4 */
        /*15d8*/                   DSETP.GT.AND P2, PT, |R8|, RZ, PT;                          /* 0x1a0e0000fc85dc81 */
        /*15e0*/              @!P2 DSETP.NEU.AND P1, PT, |R6|, +INF , PT;                      /* 0x1e8edffc0063e881 */
        /*15e8*/              @!P2 SEL R6, R6, R8, P1;                                         /* 0x200200002061a804 */
        /*15f0*/              @!P2 SEL R7, R7, R9, P1;                                         /* 0x200200002471e804 */
        /*15f8*/              @!P2 MOV R8, R6;                                                 /* 0x28000000180229e4 */
                                                                                               /* 0x220042b202e20287 */
        /*1608*/              @!P2 MOV R9, R7;                                                 /* 0x280000001c0269e4 */
        /*1610*/                   DMUL R20, R10, R8;                                          /* 0x5000000020a51c01 */
        /*1618*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1620*/                   DMUL R20, R10, R6;                                          /* 0x5000000018a51c01 */
        /*1628*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1630*/                   MOV R10, R20;                                               /* 0x2800000050029de4 */
        /*1638*/                   MOV R11, R21;                                               /* 0x280000005402dde4 */
                                                                                               /* 0x22c292b2b20042e7 */
        /*1648*/                   RET;                                                        /* 0x9000000000001de7 */
        /*1650*/                   ISETP.LT.U32.AND P1, PT, R7, c[0x2][0x38], PT;              /* 0x188e4800e073dc03 */
        /*1658*/                   MOV R7, R17;                                                /* 0x280000004401dde4 */
        /*1660*/                   PBK 0x17a8;                                                 /* 0x6800000500001c07 */
        /*1668*/               @P1 BRA 0x16b8;                                                 /* 0x40000001200005e7 */
        /*1670*/                   DFMA.RM R6, R10, R6, R18;                                   /* 0x20a4000018a19c01 */
        /*1678*/                   IADD32I R10.CC, R6, 0x1;                                    /* 0x0c00000004629c02 */
                                                                                               /* 0x22d2e04202d29287 */
        /*1688*/                   IADD.X R11, RZ, R7;                                         /* 0x480000001ff2dc43 */
        /*1690*/                   DFMA.RP R8, -R6, R10, R8;                                   /* 0x2110000028621e01 */
        /*1698*/                   DSETP.GT.AND P1, PT, R8, RZ, PT;                            /* 0x1a0e0000fc83dc01 */
        /*16a0*/                   SEL R6, R10, R6, P1;                                        /* 0x2002000018a19c04 */
        /*16a8*/                   SEL R7, R11, R7, P1;                                        /* 0x200200001cb1dc04 */
        /*16b0*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*16b8*/                   DSETP.EQU.AND P1, PT, R8, RZ, PT;                           /* 0x1d0e0000fc83dc01 */
                                                                                               /* 0x22b2c2b20042c2b7 */
        /*16c8*/               @P1 BRA 0x1798;                                                 /* 0x40000003200005e7 */
        /*16d0*/                   ISETP.LT.AND P1, PT, R9, RZ, PT;                            /* 0x188e0000fc93dc23 */
        /*16d8*/               @P1 MOV32I R6, 0x0;                                             /* 0x18000000000185e2 */
        /*16e0*/               @P1 MOV32I R7, 0xfff80000;                                      /* 0x1bffe0000001c5e2 */
        /*16e8*/               @P1 BRK;                                                        /* 0xa8000000000005e7 */
        /*16f0*/                   ISETP.GT.AND P1, PT, R9, c[0x2][0x3c], PT;                  /* 0x1a0e4800f093dc23 */
        /*16f8*/               @P1 BRA 0x1798;                                                 /* 0x40000002600005e7 */
                                                                                               /* 0x2292928042804207 */
        /*1708*/                   DMUL R6, R8, 8.11296384146066816958e+31;                    /* 0x5000d1a400819c01 */
        /*1710*/                   MOV R8, RZ;                                                 /* 0x28000000fc021de4 */
        /*1718*/                   MOV32I R20, 0x0;                                            /* 0x1800000000051de2 */
        /*1720*/                   MUFU.RSQ64H R9, R7;                                         /* 0xc80000001c725c00 */
        /*1728*/                   MOV32I R21, 0x3fe00000;                                     /* 0x18ff800000055de2 */
        /*1730*/                   DMUL R10, R8, R8;                                           /* 0x5000000020829c01 */
        /*1738*/                   DFMA R10, -R6, R10, c[0x2][0x0];                            /* 0x2014880000629e01 */
                                                                                               /* 0x2292928202929237 */
        /*1748*/                   DFMA R18, R10, 0.375, R20;                                  /* 0x2028cff600a49c01 */
        /*1750*/                   DMUL R20, R8, R10;                                          /* 0x5000000028851c01 */
        /*1758*/                   DFMA R8, R18, R20, R8;                                      /* 0x2010000051221c01 */
        /*1760*/                   DMUL R10, R6, R8;                                           /* 0x5000000020629c01 */
        /*1768*/                   IADD32I R9, R9, -0x100000;                                  /* 0x0bffc00000925c02 */
        /*1770*/                   DFMA R6, -R10, R10, R6;                                     /* 0x200c000028a19e01 */
        /*1778*/                   DFMA R6, R8, R6, R10;                                       /* 0x2014000018819c01 */
                                                                                               /* 0x22e20042b202e047 */
        /*1788*/                   IADD32I R7, R7, -0x3500000;                                 /* 0x0bf2c0000071dc02 */
        /*1790*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1798*/                   DADD R6, R8, R8;                                            /* 0x4800000020819c01 */
        /*17a0*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*17a8*/                   MOV R30, R6;                                                /* 0x2800000018079de4 */
        /*17b0*/                   MOV R31, R7;                                                /* 0x280000001c07dde4 */
        /*17b8*/                   RET;                                                        /* 0x9000000000001de7 */
        /*17c0*/                   BRA 0x17c0;                                                 /* 0x4003ffffe0001de7 */
        /*17c8*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*17d0*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*17d8*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*17e0*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*17e8*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*17f0*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*17f8*/                   NOP;                                                        /* 0x4000000000001de4 */
		.....................................................................................................................................................................


		Function : _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_11reset_forceEvEEE4callEPS4_mEUlS8_mE_JS8_mEEEvT_DpT0_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                      /* 0x22c2b20282823307 */
        /*0008*/                   MOV R1, c[0x0][0x44];                              /* 0x2800400110005de4 */
        /*0010*/                   S2R R0, SR_TID.X;                                  /* 0x2c00000084001c04 */
        /*0018*/                   S2R R3, SR_CTAID.X;                                /* 0x2c0000009400dc04 */
        /*0020*/                   IMAD R0, R3, c[0x0][0x28], R0;                     /* 0x20004000a0301ca3 */
        /*0028*/                   ISUB RZ.CC, R0, c[0x0][0x150];                     /* 0x48014005400fdd03 */
        /*0030*/                   SHR R2, R0, 0x1f;                                  /* 0x5800c0007c009c23 */
        /*0038*/                   ISETP.GE.U32.X.AND P0, PT, R2, c[0x0][0x154], PT;  /* 0x1b0e40055021dc43 */
                                                                                      /* 0x2200423042c042e7 */
        /*0048*/               @P0 EXIT;                                              /* 0x80000000000001e7 */
        /*0050*/                   IADD R4.CC, R0, c[0x0][0x148];                     /* 0x4801400520011c03 */
        /*0058*/                   MOV32I R0, 0x3;                                    /* 0x180000000c001de2 */
        /*0060*/                   IADD.X R2, R2, c[0x0][0x14c];                      /* 0x4800400530209c43 */
        /*0068*/                   SHL R3, R4, -0x1d;                                 /* 0x6000ffff8c40dc03 */
        /*0070*/                   IADD32I RZ.CC, R0, -0x20;                          /* 0x0fffffff800fdc02 */
        /*0078*/                   SHR.U32 R5, R4, 0x1d;                              /* 0x5800c00074415c03 */
                                                                                      /* 0x228042c042027237 */
        /*0088*/                   MOV32I R0, 0x0;                                    /* 0x1800000000001de2 */
        /*0090*/                   SHL R2, R2, 0x3;                                   /* 0x6000c0000c209c03 */
        /*0098*/                   CSETP.SFT.AND P0, PT, CC, PT;                      /* 0x040e00000151dc04 */
        /*00a0*/                   IMAD.U32.U32 R4.CC, R4, 0x8, R0;                   /* 0x2001c00020411c03 */
        /*00a8*/                   MOV32I R0, 0x0;                                    /* 0x1800000000001de2 */
        /*00b0*/               @P0 IADD R3, R2, R5;                                   /* 0x480000001420c003 */
        /*00b8*/                   MOV R2, R4;                                        /* 0x2800000010009de4 */
                                                                                      /* 0x20000002f042e287 */
        /*00c8*/                   IADD.X R3, R0, R3;                                 /* 0x480000000c00dc43 */
        /*00d0*/                   ST.E.64 [R2+0x4e210], RZ;                          /* 0x94001388402fdca5 */
        /*00d8*/                   ST.E.64 [R2+0x5dc10], RZ;                          /* 0x94001770402fdca5 */
        /*00e0*/                   EXIT;                                              /* 0x8000000000001de7 */
        /*00e8*/                   BRA 0xe8;                                          /* 0x4003ffffe0001de7 */
        /*00f0*/                   NOP;                                               /* 0x4000000000001de4 */
        /*00f8*/                   NOP;                                               /* 0x4000000000001de4 */
		..........................................................................................................................................................


		Function : _Z22codengen_simple_updateP4Bodyd
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                              /* 0x22404230428042b7 */
        /*0008*/                   MOV R1, c[0x0][0x44];                                      /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R0, 0x3;                                            /* 0x180000000c001de2 */
        /*0018*/                   MOV R2, c[0x0][0x144];                                     /* 0x2800400510009de4 */
        /*0020*/                   MOV R5, c[0x0][0x140];                                     /* 0x2800400500015de4 */
        /*0028*/                   IADD32I RZ.CC, R0, -0x20;                                  /* 0x0fffffff800fdc02 */
        /*0030*/                   MOV32I R0, 0x0;                                            /* 0x1800000000001de2 */
        /*0038*/                   SHL R2, R2, 0x3;                                           /* 0x6000c0000c209c03 */
                                                                                              /* 0x228042c042023237 */
        /*0048*/                   SHL R3, R5, -0x1d;                                         /* 0x6000ffff8c50dc03 */
        /*0050*/                   CSETP.SFT.AND P0, PT, CC, PT;                              /* 0x040e00000151dc04 */
        /*0058*/                   SHR.U32 R4, R5, 0x1d;                                      /* 0x5800c00074511c03 */
        /*0060*/                   IMAD.U32.U32 R5.CC, R5, 0x8, R0;                           /* 0x2001c00020515c03 */
        /*0068*/                   MOV32I R0, 0x0;                                            /* 0x1800000000001de2 */
        /*0070*/               @P0 IADD R3, R2, R4;                                           /* 0x480000001020c003 */
        /*0078*/                   MOV R2, R5;                                                /* 0x2800000014009de4 */
                                                                                              /* 0x23f240423042b047 */
        /*0088*/                   IADD.X R3, R0, R3;                                         /* 0x480000000c00dc43 */
        /*0090*/                   MOV32I R0, 0xfffffc18;                                     /* 0x1bfffff060001de2 */
        /*0098*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*00a0*/                   IADD32I R0, R0, 0x5;                                       /* 0x0800000014001c02 */
        /*00a8*/                   LD.E.64 R4, [R2+0x4e210];                                  /* 0x8400138840211ca5 */
        /*00b0*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*00b8*/                   ISETP.NE.AND P0, PT, R0, RZ, PT;                           /* 0x1a8e0000fc01dc23 */
                                                                                              /* 0x2292929252329287 */
        /*00c8*/                   MUFU.RCP64H R11, R7;                                       /* 0xc80000001872dc00 */
        /*00d0*/                   DFMA R8, -R6, R10, c[0x2][0x0];                            /* 0x2014880000621e01 */
        /*00d8*/                   DFMA R12, R8, R8, R8;                                      /* 0x2010000020831c01 */
        /*00e0*/                   DMUL R8, R4, c[0x0][0x148];                                /* 0x5000400520421c01 */
        /*00e8*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*00f0*/                   DMUL R4, R8, R10;                                          /* 0x5000000028811c01 */
        /*00f8*/                   DFMA R14, -R6, R4, R8;                                     /* 0x2010000010639e01 */
                                                                                              /* 0x22b042b2c2828207 */
        /*0108*/                   DFMA R10, R10, R14, R4;                                    /* 0x2008000038a29c01 */
        /*0110*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*0118*/                   FFMA R4, RZ, R7, R11;                                      /* 0x301600001ff11c00 */
        /*0120*/                   FSETP.GT.AND P1, PT, |R4|, c[0x2][0xc], PT;                /* 0x220e48003043dc80 */
        /*0128*/               @P1 BRA CC.NEU, 0x150;                                         /* 0x40000000800005a7 */
        /*0130*/                   MOV R4, R6;                                                /* 0x2800000018011de4 */
        /*0138*/                   MOV R5, R7;                                                /* 0x280000001c015de4 */
                                                                                              /* 0x2283f232323042b7 */
        /*0148*/                   CAL 0x978;                                                 /* 0x50000020a0011c07 */
        /*0150*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*0158*/                   MOV32I R8, 0x1;                                            /* 0x1800000004021de2 */
        /*0160*/                   LD.E.64 R6, [R2+0x5dc10];                                  /* 0x8400177040219ca5 */
        /*0168*/                   LD.E.64 R24, [R2+0x2ee10];                                 /* 0x84000bb840261ca5 */
        /*0170*/                   LD.E.64 R22, [R2+0x3e810];                                 /* 0x84000fa040259ca5 */
        /*0178*/                   MUFU.RCP64H R9, R5;                                        /* 0xc800000018525c00 */
                                                                                              /* 0x2232929292329297 */
        /*0188*/                   DFMA R12, -R4, R8, c[0x2][0x0];                            /* 0x2010880000431e01 */
        /*0190*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*0198*/                   DFMA R12, R8, R12, R8;                                     /* 0x2010000030831c01 */
        /*01a0*/                   DMUL R8, R6, c[0x0][0x148];                                /* 0x5000400520621c01 */
        /*01a8*/                   DMUL R6, R8, R12;                                          /* 0x5000000030819c01 */
        /*01b0*/                   DFMA R16, -R4, R6, R8;                                     /* 0x2010000018441e01 */
        /*01b8*/                   DFMA R16, R12, R16, R6;                                    /* 0x200c000040c41c01 */
                                                                                              /* 0x22b2b2c272004297 */
        /*01c8*/                   DADD R24, R10, R24;                                        /* 0x4800000060a61c01 */
        /*01d0*/                   FFMA R6, RZ, R5, R17;                                      /* 0x3022000017f19c00 */
        /*01d8*/                   ST.E.64 [R2+0x2ee10], R24;                                 /* 0x94000bb840261ca5 */
        /*01e0*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*01e8*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003063dc80 */
        /*01f0*/               @P1 BRA CC.NEU, 0x218;                                         /* 0x40000000800005a7 */
        /*01f8*/                   CAL 0x978;                                                 /* 0x5000001de0011c07 */
                                                                                              /* 0x223230423042b047 */
        /*0208*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
        /*0210*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*0218*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*0220*/                   DADD R16, R22, R16;                                        /* 0x4800000041641c01 */
        /*0228*/                   LD.E.64 R12, [R2+0x4e210];                                 /* 0x8400138840231ca5 */
        /*0230*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0238*/                   LD.E.64 R6, [R2+0xfa10];                                   /* 0x840003e840219ca5 */
                                                                                              /* 0x2232929283f2e2e7 */
        /*0248*/                   LD.E.64 R8, [R2+0x1f410];                                  /* 0x840007d040221ca5 */
        /*0250*/                   ST.E.64 [R2+0x3e810], R16;                                 /* 0x94000fa040241ca5 */
        /*0258*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
        /*0260*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*0268*/                   DFMA R18, -R4, R14, c[0x2][0x0];                           /* 0x201c880000449e01 */
        /*0270*/                   DFMA R18, R18, R18, R18;                                   /* 0x2024000049249c01 */
        /*0278*/                   DFMA R18, R14, R18, R14;                                   /* 0x201c000048e49c01 */
                                                                                              /* 0x2232123232929297 */
        /*0288*/                   DMUL R14, R12, c[0x0][0x148];                              /* 0x5000400520c39c01 */
        /*0290*/                   DMUL R32, R14, R18;                                        /* 0x5000000048e81c01 */
        /*0298*/                   DFMA R12, -R4, R32, R14;                                   /* 0x201c000080431e01 */
        /*02a0*/                   DFMA R18, R18, R12, R32;                                   /* 0x2040000031249c01 */
        /*02a8*/                   DFMA R6, R24, c[0x0][0x148], R6;                           /* 0x200c400521819c01 */
        /*02b0*/                   DFMA R8, R10, c[0x0][0x148], R8;                           /* 0x2010400520a21c01 */
        /*02b8*/                   FFMA R10, RZ, R5, R19;                                     /* 0x3026000017f29c00 */
                                                                                              /* 0x22b042b042c24047 */
        /*02c8*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
        /*02d0*/                   ST.E.64 [R2+0xfa10], R6;                                   /* 0x940003e840219ca5 */
        /*02d8*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*02e0*/                   ST.E.64 [R2+0x1f410], R8;                                  /* 0x940007d040221ca5 */
        /*02e8*/               @P1 BRA CC.NEU, 0x320;                                         /* 0x40000000c00005a7 */
        /*02f0*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*02f8*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
                                                                                              /* 0x2232323042b042b7 */
        /*0308*/                   CAL 0x978;                                                 /* 0x50000019a0011c07 */
        /*0310*/                   MOV R18, R10;                                              /* 0x2800000028049de4 */
        /*0318*/                   MOV R19, R11;                                              /* 0x280000002c04dde4 */
        /*0320*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*0328*/                   MOV32I R8, 0x1;                                            /* 0x1800000004021de2 */
        /*0330*/                   LD.E.64 R6, [R2+0x5dc10];                                  /* 0x8400177040219ca5 */
        /*0338*/                   LD.E.64 R24, [R2+0x2ee10];                                 /* 0x84000bb840261ca5 */
                                                                                              /* 0x22929232929283f7 */
        /*0348*/                   LD.E.64 R22, [R2+0x3e810];                                 /* 0x84000fa040259ca5 */
        /*0350*/                   MUFU.RCP64H R9, R5;                                        /* 0xc800000018525c00 */
        /*0358*/                   DFMA R10, -R4, R8, c[0x2][0x0];                            /* 0x2010880000429e01 */
        /*0360*/                   DFMA R10, R10, R10, R10;                                   /* 0x2014000028a29c01 */
        /*0368*/                   DFMA R10, R8, R10, R8;                                     /* 0x2010000028829c01 */
        /*0370*/                   DMUL R8, R6, c[0x0][0x148];                                /* 0x5000400520621c01 */
        /*0378*/                   DMUL R6, R8, R10;                                          /* 0x5000000028819c01 */
                                                                                              /* 0x22c2720042923297 */
        /*0388*/                   DFMA R12, -R4, R6, R8;                                     /* 0x2010000018431e01 */
        /*0390*/                   DFMA R10, R10, R12, R6;                                    /* 0x200c000030a29c01 */
        /*0398*/                   DADD R24, R18, R24;                                        /* 0x4800000061261c01 */
        /*03a0*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*03a8*/                   ST.E.64 [R2+0x2ee10], R24;                                 /* 0x94000bb840261ca5 */
        /*03b0*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*03b8*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003063dc80 */
                                                                                              /* 0x223230423042b2b7 */
        /*03c8*/               @P1 BRA CC.NEU, 0x3d8;                                         /* 0x40000000200005a7 */
        /*03d0*/                   CAL 0x978;                                                 /* 0x5000001680011c07 */
        /*03d8*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*03e0*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*03e8*/                   LD.E.64 R12, [R2+0x4e210];                                 /* 0x8400138840231ca5 */
        /*03f0*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*03f8*/                   LD.E.64 R6, [R2+0xfa10];                                   /* 0x840003e840219ca5 */
                                                                                              /* 0x2232929283f2e2e7 */
        /*0408*/                   LD.E.64 R8, [R2+0x1f410];                                  /* 0x840007d040221ca5 */
        /*0410*/                   ST.E.64 [R2+0x3e810], R22;                                 /* 0x94000fa040259ca5 */
        /*0418*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
        /*0420*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*0428*/                   DFMA R16, -R4, R14, c[0x2][0x0];                           /* 0x201c880000441e01 */
        /*0430*/                   DFMA R16, R16, R16, R16;                                   /* 0x2020000041041c01 */
        /*0438*/                   DFMA R16, R14, R16, R14;                                   /* 0x201c000040e41c01 */
                                                                                              /* 0x2232123232929297 */
        /*0448*/                   DMUL R14, R12, c[0x0][0x148];                              /* 0x5000400520c39c01 */
        /*0450*/                   DMUL R18, R14, R16;                                        /* 0x5000000040e49c01 */
        /*0458*/                   DFMA R12, -R4, R18, R14;                                   /* 0x201c000048431e01 */
        /*0460*/                   DFMA R16, R16, R12, R18;                                   /* 0x2024000031041c01 */
        /*0468*/                   DFMA R6, R24, c[0x0][0x148], R6;                           /* 0x200c400521819c01 */
        /*0470*/                   DFMA R8, R10, c[0x0][0x148], R8;                           /* 0x2010400520a21c01 */
        /*0478*/                   FFMA R10, RZ, R5, R17;                                     /* 0x3022000017f29c00 */
                                                                                              /* 0x22b042b042c24047 */
        /*0488*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
        /*0490*/                   ST.E.64 [R2+0xfa10], R6;                                   /* 0x940003e840219ca5 */
        /*0498*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*04a0*/                   ST.E.64 [R2+0x1f410], R8;                                  /* 0x940007d040221ca5 */
        /*04a8*/               @P1 BRA CC.NEU, 0x4e0;                                         /* 0x40000000c00005a7 */
        /*04b0*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*04b8*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
                                                                                              /* 0x2232323042b042b7 */
        /*04c8*/                   CAL 0x978;                                                 /* 0x50000012a0011c07 */
        /*04d0*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
        /*04d8*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*04e0*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*04e8*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*04f0*/                   LD.E.64 R8, [R2+0x5dc10];                                  /* 0x8400177040221ca5 */
        /*04f8*/                   LD.E.64 R6, [R2+0x2ee10];                                  /* 0x84000bb840219ca5 */
                                                                                              /* 0x22929252329283f7 */
        /*0508*/                   LD.E.64 R22, [R2+0x3e810];                                 /* 0x84000fa040259ca5 */
        /*0510*/                   MUFU.RCP64H R11, R5;                                       /* 0xc80000001852dc00 */
        /*0518*/                   DFMA R12, -R4, R10, c[0x2][0x0];                           /* 0x2014880000431e01 */
        /*0520*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*0528*/                   DMUL R8, R8, c[0x0][0x148];                                /* 0x5000400520821c01 */
        /*0530*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*0538*/                   DMUL R14, R8, R10;                                         /* 0x5000000028839c01 */
                                                                                              /* 0x22c2720042923297 */
        /*0548*/                   DFMA R12, -R4, R14, R8;                                    /* 0x2010000038431e01 */
        /*0550*/                   DFMA R10, R10, R12, R14;                                   /* 0x201c000030a29c01 */
        /*0558*/                   DADD R24, R16, R6;                                         /* 0x4800000019061c01 */
        /*0560*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*0568*/                   ST.E.64 [R2+0x2ee10], R24;                                 /* 0x94000bb840261ca5 */
        /*0570*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*0578*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003063dc80 */
                                                                                              /* 0x223230423042b2b7 */
        /*0588*/               @P1 BRA CC.NEU, 0x598;                                         /* 0x40000000200005a7 */
        /*0590*/                   CAL 0x978;                                                 /* 0x5000000f80011c07 */
        /*0598*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*05a0*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*05a8*/                   LD.E.64 R12, [R2+0x4e210];                                 /* 0x8400138840231ca5 */
        /*05b0*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*05b8*/                   LD.E.64 R6, [R2+0xfa10];                                   /* 0x840003e840219ca5 */
                                                                                              /* 0x2232929283f2e2e7 */
        /*05c8*/                   LD.E.64 R8, [R2+0x1f410];                                  /* 0x840007d040221ca5 */
        /*05d0*/                   ST.E.64 [R2+0x3e810], R22;                                 /* 0x94000fa040259ca5 */
        /*05d8*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
        /*05e0*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*05e8*/                   DFMA R16, -R4, R14, c[0x2][0x0];                           /* 0x201c880000441e01 */
        /*05f0*/                   DFMA R16, R16, R16, R16;                                   /* 0x2020000041041c01 */
        /*05f8*/                   DFMA R16, R14, R16, R14;                                   /* 0x201c000040e41c01 */
                                                                                              /* 0x2232123232929297 */
        /*0608*/                   DMUL R14, R12, c[0x0][0x148];                              /* 0x5000400520c39c01 */
        /*0610*/                   DMUL R18, R14, R16;                                        /* 0x5000000040e49c01 */
        /*0618*/                   DFMA R12, -R4, R18, R14;                                   /* 0x201c000048431e01 */
        /*0620*/                   DFMA R16, R16, R12, R18;                                   /* 0x2024000031041c01 */
        /*0628*/                   DFMA R6, R24, c[0x0][0x148], R6;                           /* 0x200c400521819c01 */
        /*0630*/                   DFMA R8, R10, c[0x0][0x148], R8;                           /* 0x2010400520a21c01 */
        /*0638*/                   FFMA R10, RZ, R5, R17;                                     /* 0x3022000017f29c00 */
                                                                                              /* 0x22b042b042c24047 */
        /*0648*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
        /*0650*/                   ST.E.64 [R2+0xfa10], R6;                                   /* 0x940003e840219ca5 */
        /*0658*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*0660*/                   ST.E.64 [R2+0x1f410], R8;                                  /* 0x940007d040221ca5 */
        /*0668*/               @P1 BRA CC.NEU, 0x6a0;                                         /* 0x40000000c00005a7 */
        /*0670*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0678*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
                                                                                              /* 0x2232323042b042b7 */
        /*0688*/                   CAL 0x978;                                                 /* 0x5000000ba0011c07 */
        /*0690*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
        /*0698*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*06a0*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*06a8*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*06b0*/                   LD.E.64 R8, [R2+0x5dc10];                                  /* 0x8400177040221ca5 */
        /*06b8*/                   LD.E.64 R6, [R2+0x2ee10];                                  /* 0x84000bb840219ca5 */
                                                                                              /* 0x22929252329283f7 */
        /*06c8*/                   LD.E.64 R22, [R2+0x3e810];                                 /* 0x84000fa040259ca5 */
        /*06d0*/                   MUFU.RCP64H R11, R5;                                       /* 0xc80000001852dc00 */
        /*06d8*/                   DFMA R12, -R4, R10, c[0x2][0x0];                           /* 0x2014880000431e01 */
        /*06e0*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*06e8*/                   DMUL R8, R8, c[0x0][0x148];                                /* 0x5000400520821c01 */
        /*06f0*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*06f8*/                   DMUL R14, R8, R10;                                         /* 0x5000000028839c01 */
                                                                                              /* 0x22c2720042923297 */
        /*0708*/                   DFMA R12, -R4, R14, R8;                                    /* 0x2010000038431e01 */
        /*0710*/                   DFMA R10, R10, R12, R14;                                   /* 0x201c000030a29c01 */
        /*0718*/                   DADD R24, R16, R6;                                         /* 0x4800000019061c01 */
        /*0720*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*0728*/                   ST.E.64 [R2+0x2ee10], R24;                                 /* 0x94000bb840261ca5 */
        /*0730*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*0738*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003063dc80 */
                                                                                              /* 0x223230423042b2b7 */
        /*0748*/               @P1 BRA CC.NEU, 0x758;                                         /* 0x40000000200005a7 */
        /*0750*/                   CAL 0x978;                                                 /* 0x5000000880011c07 */
        /*0758*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*0760*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*0768*/                   LD.E.64 R12, [R2+0x4e210];                                 /* 0x8400138840231ca5 */
        /*0770*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0778*/                   LD.E.64 R6, [R2+0xfa10];                                   /* 0x840003e840219ca5 */
                                                                                              /* 0x2232929283f2e2e7 */
        /*0788*/                   LD.E.64 R8, [R2+0x1f410];                                  /* 0x840007d040221ca5 */
        /*0790*/                   ST.E.64 [R2+0x3e810], R22;                                 /* 0x94000fa040259ca5 */
        /*0798*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
        /*07a0*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*07a8*/                   DFMA R16, -R4, R14, c[0x2][0x0];                           /* 0x201c880000441e01 */
        /*07b0*/                   DFMA R16, R16, R16, R16;                                   /* 0x2020000041041c01 */
        /*07b8*/                   DFMA R16, R14, R16, R14;                                   /* 0x201c000040e41c01 */
                                                                                              /* 0x2232123232929297 */
        /*07c8*/                   DMUL R14, R12, c[0x0][0x148];                              /* 0x5000400520c39c01 */
        /*07d0*/                   DMUL R18, R14, R16;                                        /* 0x5000000040e49c01 */
        /*07d8*/                   DFMA R12, -R4, R18, R14;                                   /* 0x201c000048431e01 */
        /*07e0*/                   DFMA R16, R16, R12, R18;                                   /* 0x2024000031041c01 */
        /*07e8*/                   DFMA R6, R24, c[0x0][0x148], R6;                           /* 0x200c400521819c01 */
        /*07f0*/                   DFMA R8, R10, c[0x0][0x148], R8;                           /* 0x2010400520a21c01 */
        /*07f8*/                   FFMA R10, RZ, R5, R17;                                     /* 0x3022000017f29c00 */
                                                                                              /* 0x22b042b042c24047 */
        /*0808*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
        /*0810*/                   ST.E.64 [R2+0xfa10], R6;                                   /* 0x940003e840219ca5 */
        /*0818*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*0820*/                   ST.E.64 [R2+0x1f410], R8;                                  /* 0x940007d040221ca5 */
        /*0828*/               @P1 BRA CC.NEU, 0x860;                                         /* 0x40000000c00005a7 */
        /*0830*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0838*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
                                                                                              /* 0x2232323042b042b7 */
        /*0848*/                   CAL 0x978;                                                 /* 0x50000004a0011c07 */
        /*0850*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
        /*0858*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*0860*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*0868*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*0870*/                   LD.E.64 R8, [R2+0x5dc10];                                  /* 0x8400177040221ca5 */
        /*0878*/                   LD.E.64 R6, [R2+0x2ee10];                                  /* 0x84000bb840219ca5 */
                                                                                              /* 0x22929252329283f7 */
        /*0888*/                   LD.E.64 R22, [R2+0x3e810];                                 /* 0x84000fa040259ca5 */
        /*0890*/                   MUFU.RCP64H R11, R5;                                       /* 0xc80000001852dc00 */
        /*0898*/                   DFMA R12, -R4, R10, c[0x2][0x0];                           /* 0x2014880000431e01 */
        /*08a0*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*08a8*/                   DMUL R8, R8, c[0x0][0x148];                                /* 0x5000400520821c01 */
        /*08b0*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*08b8*/                   DMUL R14, R8, R10;                                         /* 0x5000000028839c01 */
                                                                                              /* 0x22c2720042923297 */
        /*08c8*/                   DFMA R12, -R4, R14, R8;                                    /* 0x2010000038431e01 */
        /*08d0*/                   DFMA R10, R10, R12, R14;                                   /* 0x201c000030a29c01 */
        /*08d8*/                   DADD R24, R16, R6;                                         /* 0x4800000019061c01 */
        /*08e0*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*08e8*/                   ST.E.64 [R2+0x2ee10], R24;                                 /* 0x94000bb840261ca5 */
        /*08f0*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*08f8*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003063dc80 */
                                                                                              /* 0x23f2e2e23202b2b7 */
        /*0908*/               @P1 BRA CC.NEU, 0x918;                                         /* 0x40000000200005a7 */
        /*0910*/                   CAL 0x978;                                                 /* 0x5000000180011c07 */
        /*0918*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*0920*/                   LD.E.64 R4, [R2+0xfa10];                                   /* 0x840003e840211ca5 */
        /*0928*/                   LD.E.64 R6, [R2+0x1f410];                                  /* 0x840007d040219ca5 */
        /*0930*/                   ST.E.64 [R2+0x3e810], R22;                                 /* 0x94000fa040259ca5 */
        /*0938*/                   LD.E.64 R8, [R2+0x3e810];                                  /* 0x84000fa040221ca5 */
                                                                                              /* 0x2202e2e042937297 */
        /*0948*/                   DFMA R4, R24, c[0x0][0x148], R4;                           /* 0x2008400521811c01 */
        /*0950*/                   ST.E.64 [R2+0xfa10], R4;                                   /* 0x940003e840211ca5 */
        /*0958*/                   DFMA R6, R8, c[0x0][0x148], R6;                            /* 0x200c400520819c01 */
        /*0960*/                   ST.E.64 [R2+0x1f410], R6;                                  /* 0x940007d040219ca5 */
        /*0968*/               @P0 BRA 0x98;                                                  /* 0x4003ffdca00001e7 */
        /*0970*/                   EXIT;                                                      /* 0x8000000000001de7 */
        /*0978*/                   LOP32I.AND R6, R5, 0x40000000;                             /* 0x3900000000519c02 */
                                                                                              /* 0x22028042c0427047 */
        /*0988*/                   MOV32I R12, 0x1ff00000;                                    /* 0x187fc00000031de2 */
        /*0990*/                   LOP32I.AND R13, R9, 0x7f800000;                            /* 0x39fe000000935c02 */
        /*0998*/                   ISETP.LT.U32.AND P1, PT, R6, c[0x2][0x10], PT;             /* 0x188e48004063dc03 */
        /*09a0*/                   MOV R6, RZ;                                                /* 0x28000000fc019de4 */
        /*09a8*/                   SEL R7, R12, c[0x2][0x14], !P1;                            /* 0x2012480050c1dc04 */
        /*09b0*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*09b8*/                   DMUL R10, R4, R6;                                          /* 0x5000000018429c01 */
                                                                                              /* 0x225232928042c047 */
        /*09c8*/                   ISETP.LT.U32.AND P1, PT, R13, c[0x2][0x18], PT;            /* 0x188e480060d3dc03 */
        /*09d0*/                   MOV R18, RZ;                                               /* 0x28000000fc049de4 */
        /*09d8*/                   MUFU.RCP64H R15, R11;                                      /* 0xc800000018b3dc00 */
        /*09e0*/                   SEL R19, R12, c[0x2][0x14], !P1;                           /* 0x2012480050c4dc04 */
        /*09e8*/                   DFMA R20, -R10, R14, c[0x2][0x0];                          /* 0x201c880000a51e01 */
        /*09f0*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*09f8*/                   DMUL R16, R8, R18;                                         /* 0x5000000048841c01 */
                                                                                              /* 0x2202b2d292929297 */
        /*0a08*/                   DFMA R14, R14, R20, R14;                                   /* 0x201c000050e39c01 */
        /*0a10*/                   DMUL R20, R16, R14;                                        /* 0x5000000039051c01 */
        /*0a18*/                   DFMA R10, -R10, R20, R16;                                  /* 0x2020000050a29e01 */
        /*0a20*/                   DFMA R10, R14, R10, R20;                                   /* 0x2028000028e29c01 */
        /*0a28*/                   DSETP.LEU.AND P1, PT, |R10|, RZ, PT;                       /* 0x1d8e0000fca3dc81 */
        /*0a30*/               @P1 BRA 0xbd8;                                                 /* 0x40000006800005e7 */
        /*0a38*/                   ISETP.GT.U32.AND P1, PT, R13, c[0x2][0x1c], PT;            /* 0x1a0e480070d3dc03 */
                                                                                              /* 0x22523252328202b7 */
        /*0a48*/                   DMUL R14, R6, R10;                                         /* 0x5000000028639c01 */
        /*0a50*/                   SEL R13, R12, c[0x2][0x14], !P1;                           /* 0x2012480050c35c04 */
        /*0a58*/                   MOV R12, RZ;                                               /* 0x28000000fc031de4 */
        /*0a60*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0a68*/                   DMUL R10, R12, R14;                                        /* 0x5000000038c29c01 */
        /*0a70*/                   DMUL R6, R6, R16;                                          /* 0x5000000040619c01 */
        /*0a78*/                   DFMA R12, R4, R10, -R8;                                    /* 0x2010000028431d01 */
                                                                                              /* 0x2202b2c27202d297 */
        /*0a88*/                   DFMA R14, R4, R6, -R8;                                     /* 0x2010000018439d01 */
        /*0a90*/                   DSETP.GT.AND P1, PT, |R12|, |R14|, PT;                     /* 0x1a0e000038c3dcc1 */
        /*0a98*/                   SEL R7, R7, R11, P1;                                       /* 0x200200002c71dc04 */
        /*0aa0*/                   SEL R6, R6, R10, P1;                                       /* 0x2002000028619c04 */
        /*0aa8*/                   FSETP.GTU.AND P1, PT, |R7|, 1.469367938527859385e-39, PT;  /* 0x260ec0040073dc80 */
        /*0ab0*/               @P1 BRA 0xc50;                                                 /* 0x40000006600005e7 */
        /*0ab8*/                   FSETP.LT.AND P1, PT, |R9|, 1.5046327690525280102e-36, PT;  /* 0x208ec1000093dc80 */
                                                                                              /* 0x2232829200420047 */
        /*0ac8*/                   MOV32I R15, 0x3ff00000;                                    /* 0x18ffc0000003dde2 */
        /*0ad0*/                   LOP32I.AND R10, R6, 0xfffffffe;                            /* 0x3bfffffff8629c02 */
        /*0ad8*/                   MOV R14, RZ;                                               /* 0x28000000fc039de4 */
        /*0ae0*/                   MOV R11, R7;                                               /* 0x280000001c02dde4 */
        /*0ae8*/                   LOP32I.OR R6, R6, 0x1;                                     /* 0x3800000004619c42 */
        /*0af0*/                   SEL R15, R15, c[0x2][0x20], !P1;                           /* 0x2012480080f3dc04 */
        /*0af8*/                   DMUL R4, R4, R14;                                          /* 0x5000000038411c01 */
                                                                                              /* 0x22c04282d2923297 */
        /*0b08*/                   DMUL R8, R8, R14;                                          /* 0x5000000038821c01 */
        /*0b10*/                   DFMA R12, R10, R4, -R8;                                    /* 0x2010000010a31d01 */
        /*0b18*/                   DFMA R14, R6, R4, -R8;                                     /* 0x2010000010639d01 */
        /*0b20*/                   DSETP.GT.AND P1, PT, |R12|, |R14|, PT;                     /* 0x1a0e000038c3dcc1 */
        /*0b28*/                   SEL R11, R6, R10, P1;                                      /* 0x200200002862dc04 */
        /*0b30*/                   IADD32I R13.CC, R11, 0x1;                                  /* 0x0c00000004b35c02 */
        /*0b38*/                   LOP32I.AND R10, R11, 0x1;                                  /* 0x3800000004b29c02 */
                                                                                              /* 0x2272004202b20047 */
        /*0b48*/                   ISETP.EQ.U32.AND P1, PT, R10, 0x1, PT;                     /* 0x190ec00004a3dc03 */
        /*0b50*/                   IADD.X R12, RZ, R7;                                        /* 0x480000001ff31c43 */
        /*0b58*/                   IADD32I R6.CC, R11, -0x1;                                  /* 0x0ffffffffcb19c02 */
        /*0b60*/                   SEL R13, R11, R13, !P1;                                    /* 0x2012000034b35c04 */
        /*0b68*/                   IADD32I.X R10, R7, -0x1;                                   /* 0x0bfffffffc729c42 */
        /*0b70*/                   SEL R12, R7, R12, !P1;                                     /* 0x2012000030731c04 */
        /*0b78*/                   SEL R6, R6, R11, !P1;                                      /* 0x201200002c619c04 */
                                                                                              /* 0x2202d29232820207 */
        /*0b88*/                   SEL R7, R10, R7, !P1;                                      /* 0x201200001ca1dc04 */
        /*0b90*/                   MOV R10, R13;                                              /* 0x2800000034029de4 */
        /*0b98*/                   MOV R11, R12;                                              /* 0x280000003002dde4 */
        /*0ba0*/                   DFMA R10, R4, R10, -R8;                                    /* 0x2010000028429d01 */
        /*0ba8*/                   DFMA R4, R4, R6, -R8;                                      /* 0x2010000018411d01 */
        /*0bb0*/                   DSETP.GT.AND P1, PT, |R10|, |R4|, PT;                      /* 0x1a0e000010a3dcc1 */
        /*0bb8*/                   SEL R6, R6, R13, P1;                                       /* 0x2002000034619c04 */
                                                                                              /* 0x22d28042b2d2e047 */
        /*0bc8*/                   SEL R7, R7, R12, P1;                                       /* 0x200200003071dc04 */
        /*0bd0*/                   BRA 0xc50;                                                 /* 0x40000001e0001de7 */
        /*0bd8*/                   DSETP.EQ.AND P1, PT, R10, RZ, PT;                          /* 0x190e0000fca3dc01 */
        /*0be0*/               @P1 BRA 0xc40;                                                 /* 0x40000001600005e7 */
        /*0be8*/                   MUFU.RCP64H R7, R5;                                        /* 0xc80000001851dc00 */
        /*0bf0*/                   MOV R6, RZ;                                                /* 0x28000000fc019de4 */
        /*0bf8*/                   DSETP.GT.AND P2, PT, |R6|, RZ, PT;                         /* 0x1a0e0000fc65dc81 */
                                                                                              /* 0x22e20282027202d7 */
        /*0c08*/              @!P2 DSETP.NEU.AND P1, PT, |R4|, +INF , PT;                     /* 0x1e8edffc0043e881 */
        /*0c10*/              @!P2 SEL R4, R4, R6, P1;                                        /* 0x2002000018412804 */
        /*0c18*/              @!P2 SEL R5, R5, R7, P1;                                        /* 0x200200001c516804 */
        /*0c20*/              @!P2 MOV R6, R4;                                                /* 0x280000001001a9e4 */
        /*0c28*/              @!P2 MOV R7, R5;                                                /* 0x280000001401e9e4 */
        /*0c30*/                   DMUL R6, R8, R6;                                           /* 0x5000000018819c01 */
        /*0c38*/                   BRA 0xc50;                                                 /* 0x4000000040001de7 */
                                                                                              /* 0x20000002e20042b7 */
        /*0c48*/                   DMUL R6, R8, R4;                                           /* 0x5000000010819c01 */
        /*0c50*/                   MOV R10, R6;                                               /* 0x2800000018029de4 */
        /*0c58*/                   MOV R11, R7;                                               /* 0x280000001c02dde4 */
        /*0c60*/                   RET;                                                       /* 0x9000000000001de7 */
        /*0c68*/                   BRA 0xc68;                                                 /* 0x4003ffffe0001de7 */
        /*0c70*/                   NOP;                                                       /* 0x4000000000001de4 */
        /*0c78*/                   NOP;                                                       /* 0x4000000000001de4 */
		............................................


		Function : _Z18codegen_reset_massP4Body
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                     /* 0x2240423042804307 */
        /*0008*/                   MOV R1, c[0x0][0x44];             /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R0, 0x3;                   /* 0x180000000c001de2 */
        /*0018*/                   MOV R4, c[0x0][0x144];            /* 0x2800400510011de4 */
        /*0020*/                   MOV R3, c[0x0][0x140];            /* 0x280040050000dde4 */
        /*0028*/                   IADD32I RZ.CC, R0, -0x20;         /* 0x0fffffff800fdc02 */
        /*0030*/                   MOV32I R2, 0x0;                   /* 0x1800000000009de2 */
        /*0038*/                   SHL R4, R4, 0x3;                  /* 0x6000c0000c411c03 */
                                                                     /* 0x2282827042023237 */
        /*0048*/                   SHL R0, R3, -0x1d;                /* 0x6000ffff8c301c03 */
        /*0050*/                   CSETP.SFT.AND P0, PT, CC, PT;     /* 0x040e00000151dc04 */
        /*0058*/                   SHR.U32 R5, R3, 0x1d;             /* 0x5800c00074315c03 */
        /*0060*/                   IMAD.U32.U32 R2.CC, R3, 0x8, R2;  /* 0x2005c00020309c03 */
        /*0068*/                   MOV32I R3, 0x0;                   /* 0x180000000000dde2 */
        /*0070*/               @P0 IADD R0, R4, R5;                  /* 0x4800000014400003 */
        /*0078*/                   IADD.X R3, R3, R0;                /* 0x480000000030dc43 */
                                                                     /* 0x200000000002f047 */
        /*0088*/                   ST.E.64 [R2+0x10], RZ;            /* 0x94000000402fdca5 */
        /*0090*/                   EXIT;                             /* 0x8000000000001de7 */
        /*0098*/                   BRA 0x98;                         /* 0x4003ffffe0001de7 */
        /*00a0*/                   NOP;                              /* 0x4000000000001de4 */
        /*00a8*/                   NOP;                              /* 0x4000000000001de4 */
        /*00b0*/                   NOP;                              /* 0x4000000000001de4 */
        /*00b8*/                   NOP;                              /* 0x4000000000001de4 */
		.......................................


		Function : _Z24codegen_reset_force_halfP4Body
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                     /* 0x2240423042804307 */
        /*0008*/                   MOV R1, c[0x0][0x44];             /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R0, 0x3;                   /* 0x180000000c001de2 */
        /*0018*/                   MOV R4, c[0x0][0x144];            /* 0x2800400510011de4 */
        /*0020*/                   MOV R3, c[0x0][0x140];            /* 0x280040050000dde4 */
        /*0028*/                   IADD32I RZ.CC, R0, -0x20;         /* 0x0fffffff800fdc02 */
        /*0030*/                   MOV32I R2, 0x0;                   /* 0x1800000000009de2 */
        /*0038*/                   SHL R4, R4, 0x3;                  /* 0x6000c0000c411c03 */
                                                                     /* 0x2282827042023237 */
        /*0048*/                   SHL R0, R3, -0x1d;                /* 0x6000ffff8c301c03 */
        /*0050*/                   CSETP.SFT.AND P0, PT, CC, PT;     /* 0x040e00000151dc04 */
        /*0058*/                   SHR.U32 R5, R3, 0x1d;             /* 0x5800c00074315c03 */
        /*0060*/                   IMAD.U32.U32 R2.CC, R3, 0x8, R2;  /* 0x2005c00020309c03 */
        /*0068*/                   MOV32I R3, 0x0;                   /* 0x180000000000dde2 */
        /*0070*/               @P0 IADD R0, R4, R5;                  /* 0x4800000014400003 */
        /*0078*/                   IADD.X R3, R3, R0;                /* 0x480000000030dc43 */
                                                                     /* 0x200000000002f047 */
        /*0088*/                   ST.E.64 [R2+0x4e210], RZ;         /* 0x94001388402fdca5 */
        /*0090*/                   EXIT;                             /* 0x8000000000001de7 */
        /*0098*/                   BRA 0x98;                         /* 0x4003ffffe0001de7 */
        /*00a0*/                   NOP;                              /* 0x4000000000001de4 */
        /*00a8*/                   NOP;                              /* 0x4000000000001de4 */
        /*00b0*/                   NOP;                              /* 0x4000000000001de4 */
        /*00b8*/                   NOP;                              /* 0x4000000000001de4 */
		.............................................


		Function : _Z26codegen_reset_force_manualP4Body
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                     /* 0x2240423042804307 */
        /*0008*/                   MOV R1, c[0x0][0x44];             /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R0, 0x3;                   /* 0x180000000c001de2 */
        /*0018*/                   MOV R2, c[0x0][0x144];            /* 0x2800400510009de4 */
        /*0020*/                   MOV R5, c[0x0][0x140];            /* 0x2800400500015de4 */
        /*0028*/                   IADD32I RZ.CC, R0, -0x20;         /* 0x0fffffff800fdc02 */
        /*0030*/                   MOV32I R0, 0x0;                   /* 0x1800000000001de2 */
        /*0038*/                   SHL R2, R2, 0x3;                  /* 0x6000c0000c209c03 */
                                                                     /* 0x228042c042023237 */
        /*0048*/                   SHL R3, R5, -0x1d;                /* 0x6000ffff8c50dc03 */
        /*0050*/                   CSETP.SFT.AND P0, PT, CC, PT;     /* 0x040e00000151dc04 */
        /*0058*/                   SHR.U32 R4, R5, 0x1d;             /* 0x5800c00074511c03 */
        /*0060*/                   IMAD.U32.U32 R5.CC, R5, 0x8, R0;  /* 0x2001c00020515c03 */
        /*0068*/                   MOV32I R0, 0x0;                   /* 0x1800000000001de2 */
        /*0070*/               @P0 IADD R3, R2, R4;                  /* 0x480000001020c003 */
        /*0078*/                   MOV R2, R5;                       /* 0x2800000014009de4 */
                                                                     /* 0x20000002f042e287 */
        /*0088*/                   IADD.X R3, R0, R3;                /* 0x480000000c00dc43 */
        /*0090*/                   ST.E.64 [R2+0x4e210], RZ;         /* 0x94001388402fdca5 */
        /*0098*/                   ST.E.64 [R2+0x5dc10], RZ;         /* 0x94001770402fdca5 */
        /*00a0*/                   EXIT;                             /* 0x8000000000001de7 */
        /*00a8*/                   BRA 0xa8;                         /* 0x4003ffffe0001de7 */
        /*00b0*/                   NOP;                              /* 0x4000000000001de4 */
        /*00b8*/                   NOP;                              /* 0x4000000000001de4 */
		...............................................


		Function : _Z14codegen_updateP4Bodyd
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                               /* 0x22304280420042b7 */
        /*0008*/                   MOV R1, c[0x0][0x44];                                       /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R0, 0x3;                                             /* 0x180000000c001de2 */
        /*0018*/                   MOV R2, c[0x0][0x144];                                      /* 0x2800400510009de4 */
        /*0020*/                   MOV R6, c[0x0][0x140];                                      /* 0x2800400500019de4 */
        /*0028*/                   MOV32I R8, 0x1;                                             /* 0x1800000004021de2 */
        /*0030*/                   IADD32I RZ.CC, R0, -0x20;                                   /* 0x0fffffff800fdc02 */
        /*0038*/                   SHL R3, R6, -0x1d;                                          /* 0x6000ffff8c60dc03 */
                                                                                               /* 0x2282b04202423047 */
        /*0048*/                   MOV32I R0, 0x0;                                             /* 0x1800000000001de2 */
        /*0050*/                   SHL R2, R2, 0x3;                                            /* 0x6000c0000c209c03 */
        /*0058*/                   SHR.U32 R4, R6, 0x1d;                                       /* 0x5800c00074611c03 */
        /*0060*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*0068*/                   IMAD.U32.U32 R6.CC, R6, 0x8, R0;                            /* 0x2001c00020619c03 */
        /*0070*/                   MOV32I R0, 0x0;                                             /* 0x1800000000001de2 */
        /*0078*/               @P0 IADD R3, R2, R4;                                            /* 0x480000001020c003 */
                                                                                               /* 0x2252329283f23287 */
        /*0088*/                   IADD.X R7, R0, R3;                                          /* 0x480000000c01dc43 */
        /*0090*/                   LD.E.64 R4, [R6+0x10];                                      /* 0x8400000040611ca5 */
        /*0098*/                   LD.E.64 R2, [R6+0x4e210];                                   /* 0x8400138840609ca5 */
        /*00a0*/                   MUFU.RCP64H R9, R5;                                         /* 0xc800000018525c00 */
        /*00a8*/                   DFMA R10, -R4, R8, c[0x2][0x0];                             /* 0x2010880000429e01 */
        /*00b0*/                   DFMA R10, R10, R10, R10;                                    /* 0x2014000028a29c01 */
        /*00b8*/                   DMUL R14, R2, c[0x0][0x148];                                /* 0x5000400520239c01 */
                                                                                               /* 0x22c2828202929297 */
        /*00c8*/                   DFMA R8, R8, R10, R8;                                       /* 0x2010000028821c01 */
        /*00d0*/                   DMUL R2, R14, R8;                                           /* 0x5000000020e09c01 */
        /*00d8*/                   DFMA R12, -R4, R2, R14;                                     /* 0x201c000008431e01 */
        /*00e0*/                   DFMA R8, R8, R12, R2;                                       /* 0x2004000030821c01 */
        /*00e8*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                 /* 0x160f480020ffdc80 */
        /*00f0*/                   FFMA R0, RZ, R5, R9;                                        /* 0x3012000017f01c00 */
        /*00f8*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0xc], PT;                 /* 0x220e48003001dc80 */
                                                                                               /* 0x2232b042b2b042b7 */
        /*0108*/               @P0 BRA CC.NEU, 0x138;                                          /* 0x40000000a00001a7 */
        /*0110*/                   MOV R10, R4;                                                /* 0x2800000010029de4 */
        /*0118*/                   MOV R11, R5;                                                /* 0x280000001402dde4 */
        /*0120*/                   CAL 0x2b8;                                                  /* 0x5000000640011c07 */
        /*0128*/                   MOV R8, R22;                                                /* 0x2800000058021de4 */
        /*0130*/                   MOV R9, R23;                                                /* 0x280000005c025de4 */
        /*0138*/                   LD.E.64 R10, [R6+0x10];                                     /* 0x8400000040629ca5 */
                                                                                               /* 0x2283f23232323047 */
        /*0148*/                   LD.E.64 R14, [R6+0x5dc10];                                  /* 0x8400177040639ca5 */
        /*0150*/                   MOV32I R16, 0x1;                                            /* 0x1800000004041de2 */
        /*0158*/                   LD.E.64 R18, [R6+0x2ee10];                                  /* 0x84000bb840649ca5 */
        /*0160*/                   LD.E.64 R12, [R6+0xfa10];                                   /* 0x840003e840631ca5 */
        /*0168*/                   LD.E.64 R4, [R6+0x3e810];                                   /* 0x84000fa040611ca5 */
        /*0170*/                   LD.E.64 R2, [R6+0x1f410];                                   /* 0x840007d040609ca5 */
        /*0178*/                   MUFU.RCP64H R17, R11;                                       /* 0xc800000018b45c00 */
                                                                                               /* 0x2252329292523297 */
        /*0188*/                   DFMA R20, -R10, R16, c[0x2][0x0];                           /* 0x2020880000a51e01 */
        /*0190*/                   DFMA R20, R20, R20, R20;                                    /* 0x2028000051451c01 */
        /*0198*/                   DMUL R14, R14, c[0x0][0x148];                               /* 0x5000400520e39c01 */
        /*01a0*/                   DFMA R16, R16, R20, R16;                                    /* 0x2020000051041c01 */
        /*01a8*/                   DMUL R20, R14, R16;                                         /* 0x5000000040e51c01 */
        /*01b0*/                   DFMA R22, -R10, R20, R14;                                   /* 0x201c000050a59e01 */
        /*01b8*/                   DADD R18, R8, R18;                                          /* 0x4800000048849c01 */
                                                                                               /* 0x22c042d200425237 */
        /*01c8*/                   DFMA R22, R16, R22, R20;                                    /* 0x2028000059059c01 */
        /*01d0*/                   DFMA R8, R18, c[0x0][0x148], R12;                           /* 0x2018400521221c01 */
        /*01d8*/                   FFMA R0, RZ, R11, R23;                                      /* 0x302e00002ff01c00 */
        /*01e0*/                   ST.E.64 [R6+0x2ee10], R18;                                  /* 0x94000bb840649ca5 */
        /*01e8*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                 /* 0x160f480020ffdc80 */
        /*01f0*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0xc], PT;                 /* 0x220e48003001dc80 */
        /*01f8*/                   ST.E.64 [R6+0xfa10], R8;                                    /* 0x940003e840621ca5 */
                                                                                               /* 0x22d042725232b2b7 */
        /*0208*/               @P0 BRA CC.NEU, 0x218;                                          /* 0x40000000200001a7 */
        /*0210*/                   CAL 0x2b8;                                                  /* 0x5000000280011c07 */
        /*0218*/                   DADD R4, R4, R22;                                           /* 0x4800000058411c01 */
        /*0220*/                   DSETP.GT.AND P0, PT, R8, 1, PT;                             /* 0x1a0ecffc0081dc01 */
        /*0228*/                   DFMA R2, R4, c[0x0][0x148], R2;                             /* 0x2004400520409c01 */
        /*0230*/                   ST.E.64 [R6+0x3e810], R4;                                   /* 0x94000fa040611ca5 */
        /*0238*/                   DSETP.LT.OR P0, PT, R8, -1, P0;                             /* 0x18a0effc0081dc01 */
                                                                                               /* 0x22d2b283f232b047 */
        /*0248*/                   ST.E.64 [R6+0x1f410], R2;                                   /* 0x940007d040609ca5 */
        /*0250*/              @!P0 BRA 0x278;                                                  /* 0x40000000800021e7 */
        /*0258*/                   LD.E.64 R4, [R6+0x2ee10];                                   /* 0x84000bb840611ca5 */
        /*0260*/                   LD.E.64 R2, [R6+0x1f410];                                   /* 0x840007d040609ca5 */
        /*0268*/                   F2F.F64.F64 R4, -R4;                                        /* 0x1000000011b11d04 */
        /*0270*/                   ST.E.64 [R6+0x2ee10], R4;                                   /* 0x94000bb840611ca5 */
        /*0278*/                   DSETP.GT.AND P0, PT, R2, 1, PT;                             /* 0x1a0ecffc0021dc01 */
                                                                                               /* 0x2282e04283f2e2d7 */
        /*0288*/                   DSETP.LT.OR P0, PT, R2, -1, P0;                             /* 0x18a0effc0021dc01 */
        /*0290*/              @!P0 EXIT;                                                       /* 0x80000000000021e7 */
        /*0298*/                   LD.E.64 R2, [R6+0x3e810];                                   /* 0x84000fa040609ca5 */
        /*02a0*/                   F2F.F64.F64 R2, -R2;                                        /* 0x1000000009b09d04 */
        /*02a8*/                   ST.E.64 [R6+0x3e810], R2;                                   /* 0x94000fa040609ca5 */
        /*02b0*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*02b8*/                   LOP32I.AND R0, R11, 0x40000000;                             /* 0x3900000000b01c02 */
                                                                                               /* 0x22028042b0420047 */
        /*02c8*/                   ISETP.LT.U32.AND P0, PT, R0, c[0x2][0x10], PT;              /* 0x188e48004001dc03 */
        /*02d0*/                   LOP32I.AND R26, R15, 0x7f800000;                            /* 0x39fe000000f69c02 */
        /*02d8*/                   MOV32I R0, 0x1ff00000;                                      /* 0x187fc00000001de2 */
        /*02e0*/                   MOV R12, RZ;                                                /* 0x28000000fc031de4 */
        /*02e8*/                   SEL R13, R0, c[0x2][0x14], !P0;                             /* 0x2010480050035c04 */
        /*02f0*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
        /*02f8*/                   DMUL R16, R10, R12;                                         /* 0x5000000030a41c01 */
                                                                                               /* 0x225232928042c047 */
        /*0308*/                   ISETP.LT.U32.AND P0, PT, R26, c[0x2][0x18], PT;             /* 0x188e480061a1dc03 */
        /*0310*/                   MOV R24, RZ;                                                /* 0x28000000fc061de4 */
        /*0318*/                   MUFU.RCP64H R19, R17;                                       /* 0xc80000001914dc00 */
        /*0320*/                   SEL R25, R0, c[0x2][0x14], !P0;                             /* 0x2010480050065c04 */
        /*0328*/                   DFMA R20, -R16, R18, c[0x2][0x0];                           /* 0x2024880001051e01 */
        /*0330*/                   DFMA R22, R20, R20, R20;                                    /* 0x2028000051459c01 */
        /*0338*/                   DMUL R20, R14, R24;                                         /* 0x5000000060e51c01 */
                                                                                               /* 0x2202b2d292929297 */
        /*0348*/                   DFMA R18, R18, R22, R18;                                    /* 0x2024000059249c01 */
        /*0350*/                   DMUL R22, R20, R18;                                         /* 0x5000000049459c01 */
        /*0358*/                   DFMA R16, -R16, R22, R20;                                   /* 0x2028000059041e01 */
        /*0360*/                   DFMA R22, R18, R16, R22;                                    /* 0x202c000041259c01 */
        /*0368*/                   DSETP.LEU.AND P0, PT, |R22|, RZ, PT;                        /* 0x1d8e0000fd61dc81 */
        /*0370*/               @P0 BRA 0x528;                                                  /* 0x40000006c00001e7 */
        /*0378*/                   ISETP.GT.U32.AND P0, PT, R26, c[0x2][0x1c], PT;             /* 0x1a0e480071a1dc03 */
                                                                                               /* 0x225232523282b047 */
        /*0388*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*0390*/                   DMUL R18, R12, R22;                                         /* 0x5000000058c49c01 */
        /*0398*/                   SEL R17, R0, c[0x2][0x14], !P0;                             /* 0x2010480050045c04 */
        /*03a0*/                   DMUL R22, R22, R16;                                         /* 0x5000000041659c01 */
        /*03a8*/                   DMUL R16, R16, R18;                                         /* 0x5000000049041c01 */
        /*03b0*/                   DMUL R12, R12, R22;                                         /* 0x5000000058c31c01 */
        /*03b8*/                   DFMA R18, R10, R16, -R14;                                   /* 0x201c000040a49d01 */
                                                                                               /* 0x2202b2c27202d297 */
        /*03c8*/                   DFMA R20, R10, R12, -R14;                                   /* 0x201c000030a51d01 */
        /*03d0*/                   DSETP.GT.AND P0, PT, |R18|, |R20|, PT;                      /* 0x1a0e00005121dcc1 */
        /*03d8*/                   SEL R13, R13, R17, P0;                                      /* 0x2000000044d35c04 */
        /*03e0*/                   SEL R12, R12, R16, P0;                                      /* 0x2000000040c31c04 */
        /*03e8*/                   FSETP.GTU.AND P0, PT, |R13|, 1.469367938527859385e-39, PT;  /* 0x260ec00400d1dc80 */
        /*03f0*/               @P0 BRA 0x5a0;                                                  /* 0x40000006a00001e7 */
        /*03f8*/                   FSETP.LT.AND P0, PT, |R15|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000f1dc80 */
                                                                                               /* 0x2232804270423047 */
        /*0408*/                   MOV32I R17, 0x3ff00000;                                     /* 0x18ffc00000045de2 */
        /*0410*/                   LOP32I.AND R0, R12, 0xfffffffe;                             /* 0x3bfffffff8c01c02 */
        /*0418*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*0420*/                   LOP32I.OR R20, R12, 0x1;                                    /* 0x3800000004c51c42 */
        /*0428*/                   SEL R17, R17, c[0x2][0x20], !P0;                            /* 0x2010480081145c04 */
        /*0430*/                   MOV R12, R0;                                                /* 0x2800000000031de4 */
        /*0438*/                   DMUL R10, R10, R16;                                         /* 0x5000000040a29c01 */
                                                                                               /* 0x2282d29232720207 */
        /*0448*/                   DMUL R14, R14, R16;                                         /* 0x5000000040e39c01 */
        /*0450*/                   MOV R16, R20;                                               /* 0x2800000050041de4 */
        /*0458*/                   MOV R17, R13;                                               /* 0x2800000034045de4 */
        /*0460*/                   DFMA R18, R12, R10, -R14;                                   /* 0x201c000028c49d01 */
        /*0468*/                   DFMA R16, R16, R10, -R14;                                   /* 0x201c000029041d01 */
        /*0470*/                   DSETP.GT.AND P0, PT, |R18|, |R16|, PT;                      /* 0x1a0e00004121dcc1 */
        /*0478*/                   SEL R20, R20, R0, P0;                                       /* 0x2000000001451c04 */
                                                                                               /* 0x220042c20042c047 */
        /*0488*/                   IADD32I R19.CC, R20, 0x1;                                   /* 0x0c0000000544dc02 */
        /*0490*/                   LOP32I.AND R12, R20, 0x1;                                   /* 0x3800000005431c02 */
        /*0498*/                   ISETP.EQ.U32.AND P0, PT, R12, 0x1, PT;                      /* 0x190ec00004c1dc03 */
        /*04a0*/                   IADD.X R0, RZ, R13;                                         /* 0x4800000037f01c43 */
        /*04a8*/                   IADD32I R16.CC, R20, -0x1;                                  /* 0x0ffffffffd441c02 */
        /*04b0*/                   SEL R19, R20, R19, !P0;                                     /* 0x201000004d44dc04 */
        /*04b8*/                   IADD32I.X R17, R13, -0x1;                                   /* 0x0bfffffffcd45c42 */
                                                                                               /* 0x2292328202026207 */
        /*04c8*/                   SEL R0, R13, R0, !P0;                                       /* 0x2010000000d01c04 */
        /*04d0*/                   SEL R16, R16, R20, !P0;                                     /* 0x2010000051041c04 */
        /*04d8*/                   SEL R17, R17, R13, !P0;                                     /* 0x2010000035145c04 */
        /*04e0*/                   MOV R12, R19;                                               /* 0x280000004c031de4 */
        /*04e8*/                   MOV R13, R0;                                                /* 0x2800000000035de4 */
        /*04f0*/                   DFMA R12, R10, R12, -R14;                                   /* 0x201c000030a31d01 */
        /*04f8*/                   DFMA R10, R10, R16, -R14;                                   /* 0x201c000040a29d01 */
                                                                                               /* 0x2202b2d2e04202d7 */
        /*0508*/                   DSETP.GT.AND P0, PT, |R12|, |R10|, PT;                      /* 0x1a0e000028c1dcc1 */
        /*0510*/                   SEL R12, R16, R19, P0;                                      /* 0x200000004d031c04 */
        /*0518*/                   SEL R13, R17, R0, P0;                                       /* 0x2000000001135c04 */
        /*0520*/                   BRA 0x5a0;                                                  /* 0x40000001e0001de7 */
        /*0528*/                   DSETP.EQ.AND P0, PT, R22, RZ, PT;                           /* 0x190e0000fd61dc01 */
        /*0530*/               @P0 BRA 0x598;                                                  /* 0x40000001800001e7 */
        /*0538*/                   MUFU.RCP64H R13, R11;                                       /* 0xc800000018b35c00 */
                                                                                               /* 0x2282027202d2d287 */
        /*0548*/                   MOV R12, RZ;                                                /* 0x28000000fc031de4 */
        /*0550*/                   DSETP.GT.AND P1, PT, |R12|, RZ, PT;                         /* 0x1a0e0000fcc3dc81 */
        /*0558*/              @!P1 DSETP.NEU.AND P0, PT, |R10|, +INF , PT;                     /* 0x1e8edffc00a1e481 */
        /*0560*/              @!P1 SEL R0, R10, R12, P0;                                       /* 0x2000000030a02404 */
        /*0568*/              @!P1 SEL R11, R11, R13, P0;                                      /* 0x2000000034b2e404 */
        /*0570*/              @!P1 MOV R12, R0;                                                /* 0x28000000000325e4 */
        /*0578*/              @!P1 MOV R13, R11;                                               /* 0x280000002c0365e4 */
                                                                                               /* 0x2002e20042b2e207 */
        /*0588*/                   DMUL R12, R14, R12;                                         /* 0x5000000030e31c01 */
        /*0590*/                   BRA 0x5a0;                                                  /* 0x4000000020001de7 */
        /*0598*/                   DMUL R12, R14, R10;                                         /* 0x5000000028e31c01 */
        /*05a0*/                   MOV R22, R12;                                               /* 0x2800000030059de4 */
        /*05a8*/                   MOV R23, R13;                                               /* 0x280000003405dde4 */
        /*05b0*/                   RET;                                                        /* 0x9000000000001de7 */
        /*05b8*/                   BRA 0x5b8;                                                  /* 0x4003ffffe0001de7 */
		....................................


		Function : _Z17codegen_add_forceP4BodyS0_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                               /* 0x2282e2c2c28042b7 */
        /*0008*/                   MOV R1, c[0x0][0x44];                                       /* 0x2800400110005de4 */
        /*0010*/                   MOV R12, c[0x0][0x140];                                     /* 0x2800400500031de4 */
        /*0018*/                   MOV R3, c[0x0][0x144];                                      /* 0x280040051000dde4 */
        /*0020*/                   ISUB RZ.CC, R12, c[0x0][0x148];                             /* 0x4801400520cfdd03 */
        /*0028*/                   ISETP.EQ.X.AND P0, PT, R3, c[0x0][0x14c], PT;               /* 0x190e40053031dc63 */
        /*0030*/               @P0 EXIT;                                                       /* 0x80000000000001e7 */
        /*0038*/                   MOV32I R0, 0x3;                                             /* 0x180000000c001de2 */
                                                                                               /* 0x2242304220420047 */
        /*0048*/                   IADD32I RZ.CC, R0, -0x20;                                   /* 0x0fffffff800fdc02 */
        /*0050*/                   SHL R3, R3, 0x3;                                            /* 0x6000c0000c30dc03 */
        /*0058*/                   MOV32I R2, 0x0;                                             /* 0x1800000000009de2 */
        /*0060*/                   MOV R5, c[0x0][0x14c];                                      /* 0x2800400530015de4 */
        /*0068*/                   SHL R0, R12, -0x1d;                                         /* 0x6000ffff8cc01c03 */
        /*0070*/                   MOV R7, c[0x0][0x148];                                      /* 0x280040052001dde4 */
        /*0078*/                   SHR.U32 R4, R12, 0x1d;                                      /* 0x5800c00074c11c03 */
                                                                                               /* 0x2230428220420047 */
        /*0088*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*0090*/                   MOV32I R14, 0x0;                                            /* 0x1800000000039de2 */
        /*0098*/                   IMAD.U32.U32 R12.CC, R12, 0x8, R2;                          /* 0x2005c00020c31c03 */
        /*00a0*/                   MOV32I R15, 0x3fe00000;                                     /* 0x18ff80000003dde2 */
        /*00a8*/                   SHL R5, R5, 0x3;                                            /* 0x6000c0000c515c03 */
        /*00b0*/               @P0 IADD R0, R3, R4;                                            /* 0x4800000010300003 */
        /*00b8*/                   SHL R4, R7, -0x1d;                                          /* 0x6000ffff8c711c03 */
                                                                                               /* 0x2282c04280428047 */
        /*00c8*/                   MOV32I R3, 0x0;                                             /* 0x180000000000dde2 */
        /*00d0*/                   SHR.U32 R6, R7, 0x1d;                                       /* 0x5800c00074719c03 */
        /*00d8*/                   IADD.X R13, R0, R3;                                         /* 0x480000000c035c43 */
        /*00e0*/               @P0 IADD R4, R5, R6;                                            /* 0x4800000018510003 */
        /*00e8*/                   IMAD.U32.U32 R2.CC, R7, 0x8, R2;                            /* 0x2005c00020709c03 */
        /*00f0*/                   LD.E.64 R10, [R12+0x1f410];                                 /* 0x840007d040c29ca5 */
        /*00f8*/                   IADD.X R3, R3, R4;                                          /* 0x480000001030dc43 */
                                                                                               /* 0x2292923273f23237 */
        /*0108*/                   LD.E.64 R6, [R2+0x1f410];                                   /* 0x840007d040219ca5 */
        /*0110*/                   LD.E.64 R24, [R12+0xfa10];                                  /* 0x840003e840c61ca5 */
        /*0118*/                   LD.E.64 R4, [R2+0xfa10];                                    /* 0x840003e840211ca5 */
        /*0120*/                   DADD R10, -R10, R6;                                         /* 0x4800000018a29e01 */
        /*0128*/                   DADD R24, -R24, R4;                                         /* 0x4800000011861e01 */
        /*0130*/                   DMUL R4, R10, R10;                                          /* 0x5000000028a11c01 */
        /*0138*/                   DFMA R4, R24, R24, R4;                                      /* 0x2008000061811c01 */
                                                                                               /* 0x2292329292828047 */
        /*0148*/                   IADD32I R0, R5, -0x3500000;                                 /* 0x0bf2c00000501c02 */
        /*0150*/                   MUFU.RSQ64H R7, R5;                                         /* 0xc80000001c51dc00 */
        /*0158*/                   MOV R6, R0;                                                 /* 0x2800000000019de4 */
        /*0160*/                   DMUL R8, R6, R6;                                            /* 0x5000000018621c01 */
        /*0168*/                   DFMA R8, -R4, R8, c[0x2][0x0];                              /* 0x2010880000421e01 */
        /*0170*/                   DFMA R16, R8, 0.375, R14;                                   /* 0x201ccff600841c01 */
        /*0178*/                   DMUL R18, R6, R8;                                           /* 0x5000000020649c01 */
                                                                                               /* 0x2202928042028207 */
        /*0188*/                   DFMA R18, R16, R18, R6;                                     /* 0x200c000049049c01 */
        /*0190*/                   ISETP.LT.U32.AND P0, PT, R0, c[0x2][0x8], PT;               /* 0x188e48002001dc03 */
        /*0198*/                   DMUL R6, R4, R18;                                           /* 0x5000000048419c01 */
        /*01a0*/                   IADD32I R21, R19, -0x100000;                                /* 0x0bffc00001355c02 */
        /*01a8*/                   MOV R20, R18;                                               /* 0x2800000048051de4 */
        /*01b0*/                   DFMA R16, -R6, R6, R4;                                      /* 0x2008000018641e01 */
        /*01b8*/                   DFMA R8, R16, R20, R6;                                      /* 0x200c000051021c01 */
                                                                                               /* 0x2232b2b2020042b7 */
        /*01c8*/               @P0 BRA 0x1f8;                                                  /* 0x40000000a00001e7 */
        /*01d0*/                   MOV R8, R18;                                                /* 0x2800000048021de4 */
        /*01d8*/                   MOV R18, R6;                                                /* 0x2800000018049de4 */
        /*01e0*/                   MOV R19, R7;                                                /* 0x280000001c04dde4 */
        /*01e8*/                   MOV R9, R21;                                                /* 0x2800000054025de4 */
        /*01f0*/                   CAL 0x738;                                                  /* 0x5000001500011c07 */
        /*01f8*/                   LD.E.64 R4, [R12+0x10];                                     /* 0x8400000040c11ca5 */
                                                                                               /* 0x23f2929282820047 */
        /*0208*/                   LD.E.64 R2, [R2+0x10];                                      /* 0x8400000040209ca5 */
        /*0210*/                   DFMA R16, R8, R8, c[0x2][0x18];                             /* 0x2010880060841c01 */
        /*0218*/                   MOV32I R6, 0x1;                                             /* 0x1800000004019de2 */
        /*0220*/                   MUFU.RCP64H R7, R17;                                        /* 0xc80000001911dc00 */
        /*0228*/                   DFMA R14, -R16, R6, c[0x2][0x0];                            /* 0x200c880001039e01 */
        /*0230*/                   DFMA R18, R14, R14, R14;                                    /* 0x201c000038e49c01 */
        /*0238*/                   DFMA R18, R6, R18, R6;                                      /* 0x200c000048649c01 */
                                                                                               /* 0x2282820292929297 */
        /*0248*/                   DMUL R4, R4, c[0x2][0x10];                                  /* 0x5000480040411c01 */
        /*0250*/                   DMUL R14, R4, R2;                                           /* 0x5000000008439c01 */
        /*0258*/                   DMUL R4, R14, R18;                                          /* 0x5000000048e11c01 */
        /*0260*/                   DFMA R6, -R16, R4, R14;                                     /* 0x201c000011019e01 */
        /*0268*/                   DFMA R6, R18, R6, R4;                                       /* 0x2008000019219c01 */
        /*0270*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0xc], PT;                 /* 0x160f480030ffdc80 */
        /*0278*/                   FFMA R0, RZ, R17, R7;                                       /* 0x300e000047f01c00 */
                                                                                               /* 0x22b042b2b042b2c7 */
        /*0288*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0x20], PT;                /* 0x220e48008001dc80 */
        /*0290*/               @P0 BRA CC.NEU, 0x2c0;                                          /* 0x40000000a00001a7 */
        /*0298*/                   MOV R4, R16;                                                /* 0x2800000040011de4 */
        /*02a0*/                   MOV R5, R17;                                                /* 0x2800000044015de4 */
        /*02a8*/                   CAL 0x450;                                                  /* 0x5000000680011c07 */
        /*02b0*/                   MOV R6, R4;                                                 /* 0x2800000010019de4 */
        /*02b8*/                   MOV R7, R5;                                                 /* 0x280000001401dde4 */
                                                                                               /* 0x2292925232928047 */
        /*02c8*/                   MUFU.RCP64H R3, R9;                                         /* 0xc80000001890dc00 */
        /*02d0*/                   MOV32I R2, 0x1;                                             /* 0x1800000004009de2 */
        /*02d8*/                   DFMA R4, -R8, R2, c[0x2][0x0];                              /* 0x2004880000811e01 */
        /*02e0*/                   DFMA R4, R4, R4, R4;                                        /* 0x2008000010411c01 */
        /*02e8*/                   DMUL R22, R24, R6;                                          /* 0x5000000019859c01 */
        /*02f0*/                   DFMA R4, R2, R4, R2;                                        /* 0x2004000010211c01 */
        /*02f8*/                   DMUL R2, R22, R4;                                           /* 0x5000000011609c01 */
                                                                                               /* 0x22b2028280420297 */
        /*0308*/                   DFMA R14, -R8, R2, R22;                                     /* 0x202c000008839e01 */
        /*0310*/                   DFMA R4, R4, R14, R2;                                       /* 0x2004000038411c01 */
        /*0318*/                   FSET.GTU.AND RZ.CC, |R23|, c[0x2][0xc], PT;                 /* 0x160f4800317fdc80 */
        /*0320*/                   MOV R2, R9;                                                 /* 0x2800000024009de4 */
        /*0328*/                   FFMA R0, RZ, R9, R5;                                        /* 0x300a000027f01c00 */
        /*0330*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0x20], PT;                /* 0x220e48008001dc80 */
        /*0338*/                   MOV R0, R8;                                                 /* 0x2800000020001de4 */
                                                                                               /* 0x2232b2b2020042b7 */
        /*0348*/               @P0 BRA CC.NEU, 0x378;                                          /* 0x40000000a00001a7 */
        /*0350*/                   MOV R14, R22;                                               /* 0x2800000058039de4 */
        /*0358*/                   MOV R15, R23;                                               /* 0x280000005c03dde4 */
        /*0360*/                   MOV R4, R0;                                                 /* 0x2800000000011de4 */
        /*0368*/                   MOV R5, R2;                                                 /* 0x2800000008015de4 */
        /*0370*/                   CAL 0x450;                                                  /* 0x5000000360011c07 */
        /*0378*/                   LD.E.64 R2, [R12+0x4e210];                                  /* 0x8400138840c09ca5 */
                                                                                               /* 0x2292523292820047 */
        /*0388*/                   LD.E.64 R22, [R12+0x5dc10];                                 /* 0x8400177040c59ca5 */
        /*0390*/                   MUFU.RCP64H R19, R9;                                        /* 0xc80000001894dc00 */
        /*0398*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
        /*03a0*/                   DFMA R14, -R8, R18, c[0x2][0x0];                            /* 0x2024880000839e01 */
        /*03a8*/                   DFMA R16, R14, R14, R14;                                    /* 0x201c000038e41c01 */
        /*03b0*/                   DMUL R14, R10, R6;                                          /* 0x5000000018a39c01 */
        /*03b8*/                   DFMA R16, R18, R16, R18;                                    /* 0x2024000041241c01 */
                                                                                               /* 0x2280429233f20297 */
        /*03c8*/                   DMUL R10, R16, R14;                                         /* 0x5000000039029c01 */
        /*03d0*/                   DFMA R6, -R8, R10, R14;                                     /* 0x201c000028819e01 */
        /*03d8*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0xc], PT;                 /* 0x160f480030ffdc80 */
        /*03e0*/                   DADD R2, R4, R2;                                            /* 0x4800000008409c01 */
        /*03e8*/                   DFMA R4, R16, R6, R10;                                      /* 0x2014000019011c01 */
        /*03f0*/                   FFMA R0, RZ, R9, R5;                                        /* 0x300a000027f01c00 */
        /*03f8*/                   ST.E.64 [R12+0x4e210], R2;                                  /* 0x9400138840c09ca5 */
                                                                                               /* 0x220292b2b042b2c7 */
        /*0408*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0x20], PT;                /* 0x220e48008001dc80 */
        /*0410*/               @P0 BRA CC.NEU, 0x430;                                          /* 0x40000000600001a7 */
        /*0418*/                   MOV R4, R8;                                                 /* 0x2800000020011de4 */
        /*0420*/                   MOV R5, R9;                                                 /* 0x2800000024015de4 */
        /*0428*/                   CAL 0x450;                                                  /* 0x5000000080011c07 */
        /*0430*/                   DADD R22, R22, R4;                                          /* 0x4800000011659c01 */
        /*0438*/                   ST.E.64 [R12+0x5dc10], R22;                                 /* 0x9400177040c59ca5 */
                                                                                               /* 0x22b04200428042e7 */
        /*0448*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*0450*/                   LOP32I.AND R0, R5, 0x40000000;                              /* 0x3900000000501c02 */
        /*0458*/                   MOV R2, RZ;                                                 /* 0x28000000fc009de4 */
        /*0460*/                   ISETP.LT.U32.AND P0, PT, R0, c[0x2][0x24], PT;              /* 0x188e48009001dc03 */
        /*0468*/                   LOP32I.AND R26, R15, 0x7f800000;                            /* 0x39fe000000f69c02 */
        /*0470*/                   MOV32I R0, 0x1ff00000;                                      /* 0x187fc00000001de2 */
        /*0478*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
                                                                                               /* 0x22928042c0420287 */
        /*0488*/                   SEL R3, R0, c[0x2][0x28], !P0;                              /* 0x20104800a000dc04 */
        /*0490*/                   DMUL R16, R4, R2;                                           /* 0x5000000008441c01 */
        /*0498*/                   ISETP.LT.U32.AND P0, PT, R26, c[0x2][0x2c], PT;             /* 0x188e4800b1a1dc03 */
        /*04a0*/                   MOV R20, RZ;                                                /* 0x28000000fc051de4 */
        /*04a8*/                   MUFU.RCP64H R19, R17;                                       /* 0xc80000001914dc00 */
        /*04b0*/                   SEL R21, R0, c[0x2][0x28], !P0;                             /* 0x20104800a0055c04 */
        /*04b8*/                   DFMA R28, -R16, R18, c[0x2][0x0];                           /* 0x2024880001071e01 */
                                                                                               /* 0x22d2929292925237 */
        /*04c8*/                   DFMA R28, R28, R28, R28;                                    /* 0x2038000071c71c01 */
        /*04d0*/                   DMUL R20, R14, R20;                                         /* 0x5000000050e51c01 */
        /*04d8*/                   DFMA R18, R18, R28, R18;                                    /* 0x2024000071249c01 */
        /*04e0*/                   DMUL R30, R20, R18;                                         /* 0x5000000049479c01 */
        /*04e8*/                   DFMA R16, -R16, R30, R20;                                   /* 0x2028000079041e01 */
        /*04f0*/                   DFMA R20, R18, R16, R30;                                    /* 0x203c000041251c01 */
        /*04f8*/                   DSETP.LEU.AND P0, PT, |R20|, RZ, PT;                        /* 0x1d8e0000fd41dc81 */
                                                                                               /* 0x22523282823042b7 */
        /*0508*/               @P0 BRA 0x6a8;                                                  /* 0x40000006600001e7 */
        /*0510*/                   ISETP.GT.U32.AND P0, PT, R26, c[0x2][0x30], PT;             /* 0x1a0e4800c1a1dc03 */
        /*0518*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*0520*/                   DMUL R18, R2, R20;                                          /* 0x5000000050249c01 */
        /*0528*/                   SEL R17, R0, c[0x2][0x28], !P0;                             /* 0x20104800a0045c04 */
        /*0530*/                   DMUL R20, R20, R16;                                         /* 0x5000000041451c01 */
        /*0538*/                   DMUL R16, R16, R18;                                         /* 0x5000000049041c01 */
                                                                                               /* 0x22c27202d2925237 */
        /*0548*/                   DMUL R20, R2, R20;                                          /* 0x5000000050251c01 */
        /*0550*/                   DFMA R2, R4, R16, -R14;                                     /* 0x201c000040409d01 */
        /*0558*/                   DFMA R28, R4, R20, -R14;                                    /* 0x201c000050471d01 */
        /*0560*/                   DSETP.GT.AND P0, PT, |R2|, |R28|, PT;                       /* 0x1a0e00007021dcc1 */
        /*0568*/                   SEL R3, R21, R17, P0;                                       /* 0x200000004550dc04 */
        /*0570*/                   SEL R2, R20, R16, P0;                                       /* 0x2000000041409c04 */
        /*0578*/                   FSETP.GTU.AND P0, PT, |R3|, 1.469367938527859385e-39, PT;   /* 0x260ec0040031dc80 */
                                                                                               /* 0x22804220420042b7 */
        /*0588*/               @P0 BRA 0x720;                                                  /* 0x40000006400001e7 */
        /*0590*/                   FSETP.LT.AND P0, PT, |R15|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000f1dc80 */
        /*0598*/                   LOP32I.AND R0, R2, 0xfffffffe;                              /* 0x3bfffffff8201c02 */
        /*05a0*/                   MOV32I R17, 0x3ff00000;                                     /* 0x18ffc00000045de2 */
        /*05a8*/                   MOV R21, R3;                                                /* 0x280000000c055de4 */
        /*05b0*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*05b8*/                   LOP32I.OR R20, R2, 0x1;                                     /* 0x3800000004251c42 */
                                                                                               /* 0x22d2923292328047 */
        /*05c8*/                   SEL R17, R17, c[0x2][0x34], !P0;                            /* 0x20104800d1145c04 */
        /*05d0*/                   MOV R2, R0;                                                 /* 0x2800000000009de4 */
        /*05d8*/                   DMUL R4, R4, R16;                                           /* 0x5000000040411c01 */
        /*05e0*/                   DMUL R14, R14, R16;                                         /* 0x5000000040e39c01 */
        /*05e8*/                   DFMA R18, R20, R4, -R14;                                    /* 0x201c000011449d01 */
        /*05f0*/                   DFMA R16, R2, R4, -R14;                                     /* 0x201c000010241d01 */
        /*05f8*/                   DSETP.GT.AND P0, PT, |R16|, |R18|, PT;                      /* 0x1a0e00004901dcc1 */
                                                                                               /* 0x2232804232804287 */
        /*0608*/                   SEL R20, R20, R0, P0;                                       /* 0x2000000001451c04 */
        /*0610*/                   IADD32I R0.CC, R20, 0x1;                                    /* 0x0c00000005401c02 */
        /*0618*/                   LOP32I.AND R2, R20, 0x1;                                    /* 0x3800000005409c02 */
        /*0620*/                   ISETP.EQ.U32.AND P0, PT, R2, 0x1, PT;                       /* 0x190ec0000421dc03 */
        /*0628*/                   IADD.X R16, RZ, R3;                                         /* 0x480000000ff41c43 */
        /*0630*/                   IADD32I R18.CC, R20, -0x1;                                  /* 0x0ffffffffd449c02 */
        /*0638*/                   SEL R0, R20, R0, !P0;                                       /* 0x2010000001401c04 */
                                                                                               /* 0x2292324232320047 */
        /*0648*/                   IADD32I.X R2, R3, -0x1;                                     /* 0x0bfffffffc309c42 */
        /*0650*/                   SEL R17, R3, R16, !P0;                                      /* 0x2010000040345c04 */
        /*0658*/                   SEL R18, R18, R20, !P0;                                     /* 0x2010000051249c04 */
        /*0660*/                   MOV R16, R0;                                                /* 0x2800000000041de4 */
        /*0668*/                   SEL R19, R2, R3, !P0;                                       /* 0x201000000c24dc04 */
        /*0670*/                   DFMA R2, R4, R16, -R14;                                     /* 0x201c000040409d01 */
        /*0678*/                   DFMA R4, R4, R18, -R14;                                     /* 0x201c000048411d01 */
                                                                                               /* 0x2202b2d2e04202d7 */
        /*0688*/                   DSETP.GT.AND P0, PT, |R2|, |R4|, PT;                        /* 0x1a0e00001021dcc1 */
        /*0690*/                   SEL R2, R18, R0, P0;                                        /* 0x2000000001209c04 */
        /*0698*/                   SEL R3, R19, R17, P0;                                       /* 0x200000004530dc04 */
        /*06a0*/                   BRA 0x720;                                                  /* 0x40000001e0001de7 */
        /*06a8*/                   DSETP.EQ.AND P0, PT, R20, RZ, PT;                           /* 0x190e0000fd41dc01 */
        /*06b0*/               @P0 BRA 0x718;                                                  /* 0x40000001800001e7 */
        /*06b8*/                   MUFU.RCP64H R3, R5;                                         /* 0xc80000001850dc00 */
                                                                                               /* 0x2282027202d2d287 */
        /*06c8*/                   MOV R2, RZ;                                                 /* 0x28000000fc009de4 */
        /*06d0*/                   DSETP.GT.AND P1, PT, |R2|, RZ, PT;                          /* 0x1a0e0000fc23dc81 */
        /*06d8*/              @!P1 DSETP.NEU.AND P0, PT, |R4|, +INF , PT;                      /* 0x1e8edffc0041e481 */
        /*06e0*/              @!P1 SEL R0, R4, R2, P0;                                         /* 0x2000000008402404 */
        /*06e8*/              @!P1 SEL R4, R5, R3, P0;                                         /* 0x200000000c512404 */
        /*06f0*/              @!P1 MOV R2, R0;                                                 /* 0x280000000000a5e4 */
        /*06f8*/              @!P1 MOV R3, R4;                                                 /* 0x280000001000e5e4 */
                                                                                               /* 0x2202e20042b2e207 */
        /*0708*/                   DMUL R2, R14, R2;                                           /* 0x5000000008e09c01 */
        /*0710*/                   BRA 0x720;                                                  /* 0x4000000020001de7 */
        /*0718*/                   DMUL R2, R14, R4;                                           /* 0x5000000010e09c01 */
        /*0720*/                   MOV R4, R2;                                                 /* 0x2800000008011de4 */
        /*0728*/                   MOV R5, R3;                                                 /* 0x280000000c015de4 */
        /*0730*/                   RET;                                                        /* 0x9000000000001de7 */
        /*0738*/                   ISETP.LT.U32.AND P0, PT, R0, c[0x2][0x38], PT;              /* 0x188e4800e001dc03 */
                                                                                               /* 0x229282c292b2b047 */
        /*0748*/                   MOV R6, R16;                                                /* 0x2800000040019de4 */
        /*0750*/                   MOV R7, R17;                                                /* 0x280000004401dde4 */
        /*0758*/               @P0 BRA 0x7a8;                                                  /* 0x40000001200001e7 */
        /*0760*/                   DFMA.RM R6, R6, R8, R18;                                    /* 0x20a4000020619c01 */
        /*0768*/                   IADD32I R8.CC, R6, 0x1;                                     /* 0x0c00000004621c02 */
        /*0770*/                   IADD.X R9, RZ, R7;                                          /* 0x480000001ff25c43 */
        /*0778*/                   DFMA.RP R4, -R6, R8, R4;                                    /* 0x2108000020611e01 */
                                                                                               /* 0x22c2b2d2e04202d7 */
        /*0788*/                   DSETP.GT.AND P0, PT, R4, RZ, PT;                            /* 0x1a0e0000fc41dc01 */
        /*0790*/                   SEL R6, R8, R6, P0;                                         /* 0x2000000018819c04 */
        /*0798*/                   SEL R7, R9, R7, P0;                                         /* 0x200000001c91dc04 */
        /*07a0*/                   BRA 0x890;                                                  /* 0x40000003a0001de7 */
        /*07a8*/                   DSETP.EQU.AND P0, PT, R4, RZ, PT;                           /* 0x1d0e0000fc41dc01 */
        /*07b0*/               @P0 BRA 0x880;                                                  /* 0x40000003200001e7 */
        /*07b8*/                   ISETP.LT.AND P0, PT, R5, RZ, PT;                            /* 0x188e0000fc51dc23 */
                                                                                               /* 0x228202b2c2b20047 */
        /*07c8*/               @P0 MOV32I R6, 0x0;                                             /* 0x18000000000181e2 */
        /*07d0*/               @P0 MOV32I R7, 0xfff80000;                                      /* 0x1bffe0000001c1e2 */
        /*07d8*/               @P0 BRA 0x890;                                                  /* 0x40000002c00001e7 */
        /*07e0*/                   ISETP.GT.AND P0, PT, R5, c[0x2][0x3c], PT;                  /* 0x1a0e4800f051dc23 */
        /*07e8*/               @P0 BRA 0x880;                                                  /* 0x40000002400001e7 */
        /*07f0*/                   DMUL R4, R4, 8.11296384146066816958e+31;                    /* 0x5000d1a400411c01 */
        /*07f8*/                   MOV R6, RZ;                                                 /* 0x28000000fc019de4 */
                                                                                               /* 0x2292329282028047 */
        /*0808*/                   MUFU.RSQ64H R7, R5;                                         /* 0xc80000001c51dc00 */
        /*0810*/                   MOV32I R14, 0x0;                                            /* 0x1800000000039de2 */
        /*0818*/                   DMUL R8, R6, R6;                                            /* 0x5000000018621c01 */
        /*0820*/                   MOV32I R15, 0x3fe00000;                                     /* 0x18ff80000003dde2 */
        /*0828*/                   DFMA R8, -R4, R8, c[0x2][0x0];                              /* 0x2010880000421e01 */
        /*0830*/                   DFMA R16, R8, 0.375, R14;                                   /* 0x201ccff600841c01 */
        /*0838*/                   DMUL R18, R6, R8;                                           /* 0x5000000020649c01 */
                                                                                               /* 0x22e0429292820297 */
        /*0848*/                   DFMA R6, R16, R18, R6;                                      /* 0x200c000049019c01 */
        /*0850*/                   DMUL R8, R4, R6;                                            /* 0x5000000018421c01 */
        /*0858*/                   IADD32I R7, R7, -0x100000;                                  /* 0x0bffc0000071dc02 */
        /*0860*/                   DFMA R4, -R8, R8, R4;                                       /* 0x2008000020811e01 */
        /*0868*/                   DFMA R6, R6, R4, R8;                                        /* 0x2010000010619c01 */
        /*0870*/                   IADD32I R7, R7, -0x3500000;                                 /* 0x0bf2c0000071dc02 */
        /*0878*/                   BRA 0x890;                                                  /* 0x4000000040001de7 */
                                                                                               /* 0x20000002e20042b7 */
        /*0888*/                   DADD R6, R4, R4;                                            /* 0x4800000010419c01 */
        /*0890*/                   MOV R8, R6;                                                 /* 0x2800000018021de4 */
        /*0898*/                   MOV R9, R7;                                                 /* 0x280000001c025de4 */
        /*08a0*/                   RET;                                                        /* 0x9000000000001de7 */
        /*08a8*/                   BRA 0x8a8;                                                  /* 0x4003ffffe0001de7 */
        /*08b0*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*08b8*/                   NOP;                                                        /* 0x4000000000001de4 */
		.........................................


		Function : _Z19codegen_reset_forceP4Body
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                     /* 0x2240423042804307 */
        /*0008*/                   MOV R1, c[0x0][0x44];             /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R0, 0x3;                   /* 0x180000000c001de2 */
        /*0018*/                   MOV R2, c[0x0][0x144];            /* 0x2800400510009de4 */
        /*0020*/                   MOV R5, c[0x0][0x140];            /* 0x2800400500015de4 */
        /*0028*/                   IADD32I RZ.CC, R0, -0x20;         /* 0x0fffffff800fdc02 */
        /*0030*/                   MOV32I R0, 0x0;                   /* 0x1800000000001de2 */
        /*0038*/                   SHL R2, R2, 0x3;                  /* 0x6000c0000c209c03 */
                                                                     /* 0x228042c042023237 */
        /*0048*/                   SHL R3, R5, -0x1d;                /* 0x6000ffff8c50dc03 */
        /*0050*/                   CSETP.SFT.AND P0, PT, CC, PT;     /* 0x040e00000151dc04 */
        /*0058*/                   SHR.U32 R4, R5, 0x1d;             /* 0x5800c00074511c03 */
        /*0060*/                   IMAD.U32.U32 R5.CC, R5, 0x8, R0;  /* 0x2001c00020515c03 */
        /*0068*/                   MOV32I R0, 0x0;                   /* 0x1800000000001de2 */
        /*0070*/               @P0 IADD R3, R2, R4;                  /* 0x480000001020c003 */
        /*0078*/                   MOV R2, R5;                       /* 0x2800000014009de4 */
                                                                     /* 0x20000002f042e287 */
        /*0088*/                   IADD.X R3, R0, R3;                /* 0x480000000c00dc43 */
        /*0090*/                   ST.E.64 [R2+0x4e210], RZ;         /* 0x94001388402fdca5 */
        /*0098*/                   ST.E.64 [R2+0x5dc10], RZ;         /* 0x94001770402fdca5 */
        /*00a0*/                   EXIT;                             /* 0x8000000000001de7 */
        /*00a8*/                   BRA 0xa8;                         /* 0x4003ffffe0001de7 */
        /*00b0*/                   NOP;                              /* 0x4000000000001de4 */
        /*00b8*/                   NOP;                              /* 0x4000000000001de4 */
		........................................



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


.global .align 16 .b8 __Bodydata_buffer[448072];
.global .align 8 .u64 _ZN4ikra8executor4cuda14d_storage_sizeE;
.global .align 8 .u64 _ZN4ikra8executor4cuda19d_storage_data_headE;
.global .align 8 .u64 _ZN4ikra3soa23d_previous_storage_sizeE;

.visible .entry _Z19codegen_reset_forceP4Body(
.param .u64 _Z19codegen_reset_forceP4Body_param_0
)
{
.reg .b64 %rd<6>;


ld.param.u64 %rd1, [_Z19codegen_reset_forceP4Body_param_0];
shl.b64 %rd2, %rd1, 3;
mov.u64 %rd3, __Bodydata_buffer;
add.s64 %rd4, %rd2, %rd3;
mov.u64 %rd5, 0;
st.global.u64 [%rd4+320016], %rd5;
st.global.u64 [%rd4+384016], %rd5;
ret;
}


.visible .entry _Z17codegen_add_forceP4BodyS0_(
.param .u64 _Z17codegen_add_forceP4BodyS0__param_0,
.param .u64 _Z17codegen_add_forceP4BodyS0__param_1
)
{
.reg .pred %p<2>;
.reg .f64 %fd<24>;
.reg .b64 %rd<11>;


ld.param.u64 %rd1, [_Z17codegen_add_forceP4BodyS0__param_0];
ld.param.u64 %rd2, [_Z17codegen_add_forceP4BodyS0__param_1];
cvta.to.global.u64 %rd3, %rd2;
cvta.to.global.u64 %rd4, %rd1;
setp.eq.s64	%p1, %rd4, %rd3;
@%p1 bra BB1_2;

shl.b64 %rd5, %rd2, 3;
mov.u64 %rd6, __Bodydata_buffer;
add.s64 %rd7, %rd6, 16;
add.s64 %rd8, %rd7, %rd5;
shl.b64 %rd9, %rd1, 3;
add.s64 %rd10, %rd7, %rd9;
ld.global.f64 %fd1, [%rd10+64000];
ld.global.f64 %fd2, [%rd8+64000];
sub.f64 %fd3, %fd2, %fd1;
ld.global.f64 %fd4, [%rd10+128000];
ld.global.f64 %fd5, [%rd8+128000];
sub.f64 %fd6, %fd5, %fd4;
mul.f64 %fd7, %fd6, %fd6;
fma.rn.f64 %fd8, %fd3, %fd3, %fd7;
sqrt.rn.f64 %fd9, %fd8;
ld.global.f64 %fd10, [%rd10];
mul.f64 %fd11, %fd10, 0d3DD257B4D002790D;
ld.global.f64 %fd12, [%rd8];
mul.f64 %fd13, %fd11, %fd12;
fma.rn.f64 %fd14, %fd9, %fd9, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd15, %fd13, %fd14;
mul.f64 %fd16, %fd3, %fd15;
div.rn.f64 %fd17, %fd16, %fd9;
ld.global.f64 %fd18, [%rd10+320000];
add.f64 %fd19, %fd17, %fd18;
ld.global.f64 %fd20, [%rd10+384000];
st.global.f64 [%rd10+320000], %fd19;
mul.f64 %fd21, %fd6, %fd15;
div.rn.f64 %fd22, %fd21, %fd9;
add.f64 %fd23, %fd22, %fd20;
st.global.f64 [%rd10+384000], %fd23;

BB1_2:
ret;
}


.visible .entry _Z14codegen_updateP4Bodyd(
.param .u64 _Z14codegen_updateP4Bodyd_param_0,
.param .f64 _Z14codegen_updateP4Bodyd_param_1
)
{
.reg .pred %p<7>;
.reg .f64 %fd<25>;
.reg .b64 %rd<7>;


ld.param.u64 %rd3, [_Z14codegen_updateP4Bodyd_param_0];
ld.param.f64 %fd4, [_Z14codegen_updateP4Bodyd_param_1];
shl.b64 %rd4, %rd3, 3;
mov.u64 %rd5, __Bodydata_buffer;
add.s64 %rd6, %rd5, %rd4;
ld.global.f64 %fd5, [%rd6+320016];
mul.f64 %fd6, %fd5, %fd4;
ld.global.f64 %fd7, [%rd6+16];
div.rn.f64 %fd8, %fd6, %fd7;
add.s64 %rd1, %rd6, 192016;
ld.global.f64 %fd9, [%rd6+192016];
add.f64 %fd10, %fd8, %fd9;
ld.global.f64 %fd11, [%rd6+384016];
ld.global.f64 %fd12, [%rd6+256016];
ld.global.f64 %fd13, [%rd6+64016];
ld.global.f64 %fd14, [%rd6+128016];
st.global.f64 [%rd6+192016], %fd10;
mul.f64 %fd15, %fd11, %fd4;
fma.rn.f64 %fd16, %fd10, %fd4, %fd13;
st.global.f64 [%rd6+64016], %fd16;
ld.global.f64 %fd17, [%rd6+16];
div.rn.f64 %fd18, %fd15, %fd17;
add.f64 %fd19, %fd18, %fd12;
st.global.f64 [%rd6+256016], %fd19;
add.s64 %rd2, %rd6, 64016;
fma.rn.f64 %fd24, %fd19, %fd4, %fd14;
st.global.f64 [%rd6+128016], %fd24;
setp.lt.f64	%p1, %fd16, 0dBFF0000000000000;
setp.gt.f64	%p2, %fd16, 0d3FF0000000000000;
or.pred %p3, %p1, %p2;
@!%p3 bra BB2_2;
bra.uni BB2_1;

BB2_1:
ld.global.f64 %fd20, [%rd1];
neg.f64 %fd21, %fd20;
st.global.f64 [%rd1], %fd21;
ld.global.f64 %fd24, [%rd2+64000];

BB2_2:
setp.gt.f64	%p4, %fd24, 0d3FF0000000000000;
setp.lt.f64	%p5, %fd24, 0dBFF0000000000000;
or.pred %p6, %p5, %p4;
@!%p6 bra BB2_4;
bra.uni BB2_3;

BB2_3:
ld.global.f64 %fd22, [%rd1+64000];
neg.f64 %fd23, %fd22;
st.global.f64 [%rd1+64000], %fd23;

BB2_4:
ret;
}


.visible .entry _Z26codegen_reset_force_manualP4Body(
.param .u64 _Z26codegen_reset_force_manualP4Body_param_0
)
{
.reg .b64 %rd<6>;


ld.param.u64 %rd1, [_Z26codegen_reset_force_manualP4Body_param_0];
shl.b64 %rd2, %rd1, 3;
mov.u64 %rd3, __Bodydata_buffer;
add.s64 %rd4, %rd2, %rd3;
mov.u64 %rd5, 0;
st.global.u64 [%rd4+320016], %rd5;
st.global.u64 [%rd4+384016], %rd5;
ret;
}


.visible .entry _Z24codegen_reset_force_halfP4Body(
.param .u64 _Z24codegen_reset_force_halfP4Body_param_0
)
{
.reg .b64 %rd<6>;


ld.param.u64 %rd1, [_Z24codegen_reset_force_halfP4Body_param_0];
shl.b64 %rd2, %rd1, 3;
mov.u64 %rd3, __Bodydata_buffer;
add.s64 %rd4, %rd2, %rd3;
mov.u64 %rd5, 0;
st.global.u64 [%rd4+320016], %rd5;
ret;
}


.visible .entry _Z18codegen_reset_massP4Body(
.param .u64 _Z18codegen_reset_massP4Body_param_0
)
{
.reg .b64 %rd<6>;


ld.param.u64 %rd1, [_Z18codegen_reset_massP4Body_param_0];
shl.b64 %rd2, %rd1, 3;
mov.u64 %rd3, __Bodydata_buffer;
add.s64 %rd4, %rd3, %rd2;
mov.u64 %rd5, 0;
st.global.u64 [%rd4+16], %rd5;
ret;
}


.visible .entry _Z22codengen_simple_updateP4Bodyd(
.param .u64 _Z22codengen_simple_updateP4Bodyd_param_0,
.param .f64 _Z22codengen_simple_updateP4Bodyd_param_1
)
{
.reg .pred %p<2>;
.reg .b32 %r<5>;
.reg .f64 %fd<87>;
.reg .b64 %rd<9>;


ld.param.u64 %rd5, [_Z22codengen_simple_updateP4Bodyd_param_0];
ld.param.f64 %fd1, [_Z22codengen_simple_updateP4Bodyd_param_1];
shl.b64 %rd6, %rd5, 3;
mov.u64 %rd7, __Bodydata_buffer;
add.s64 %rd8, %rd7, %rd6;
add.s64 %rd2, %rd8, 16;
add.s64 %rd1, %rd8, 320016;
add.s64 %rd3, %rd8, 192016;
add.s64 %rd4, %rd8, 64016;
mov.u32 %r4, -1000;

BB6_1:
ld.global.f64 %fd2, [%rd1];
mul.f64 %fd3, %fd2, %fd1;
ld.global.f64 %fd4, [%rd2];
div.rn.f64 %fd5, %fd3, %fd4;
ld.global.f64 %fd6, [%rd3];
add.f64 %fd7, %fd5, %fd6;
ld.global.f64 %fd8, [%rd3+64000];
st.global.f64 [%rd3], %fd7;
ld.global.f64 %fd9, [%rd1+64000];
mul.f64 %fd10, %fd9, %fd1;
ld.global.f64 %fd11, [%rd2];
div.rn.f64 %fd12, %fd10, %fd11;
add.f64 %fd13, %fd12, %fd8;
st.global.f64 [%rd3+64000], %fd13;
ld.global.f64 %fd14, [%rd4];
fma.rn.f64 %fd15, %fd7, %fd1, %fd14;
ld.global.f64 %fd16, [%rd4+64000];
st.global.f64 [%rd4], %fd15;
ld.global.f64 %fd17, [%rd3+64000];
fma.rn.f64 %fd18, %fd17, %fd1, %fd16;
st.global.f64 [%rd4+64000], %fd18;
ld.global.f64 %fd19, [%rd1];
mul.f64 %fd20, %fd19, %fd1;
ld.global.f64 %fd21, [%rd2];
div.rn.f64 %fd22, %fd20, %fd21;
ld.global.f64 %fd23, [%rd3];
add.f64 %fd24, %fd22, %fd23;
ld.global.f64 %fd25, [%rd3+64000];
st.global.f64 [%rd3], %fd24;
ld.global.f64 %fd26, [%rd1+64000];
mul.f64 %fd27, %fd26, %fd1;
ld.global.f64 %fd28, [%rd2];
div.rn.f64 %fd29, %fd27, %fd28;
add.f64 %fd30, %fd29, %fd25;
st.global.f64 [%rd3+64000], %fd30;
ld.global.f64 %fd31, [%rd4];
fma.rn.f64 %fd32, %fd24, %fd1, %fd31;
ld.global.f64 %fd33, [%rd4+64000];
st.global.f64 [%rd4], %fd32;
ld.global.f64 %fd34, [%rd3+64000];
fma.rn.f64 %fd35, %fd34, %fd1, %fd33;
st.global.f64 [%rd4+64000], %fd35;
ld.global.f64 %fd36, [%rd1];
mul.f64 %fd37, %fd36, %fd1;
ld.global.f64 %fd38, [%rd2];
div.rn.f64 %fd39, %fd37, %fd38;
ld.global.f64 %fd40, [%rd3];
add.f64 %fd41, %fd39, %fd40;
ld.global.f64 %fd42, [%rd3+64000];
st.global.f64 [%rd3], %fd41;
ld.global.f64 %fd43, [%rd1+64000];
mul.f64 %fd44, %fd43, %fd1;
ld.global.f64 %fd45, [%rd2];
div.rn.f64 %fd46, %fd44, %fd45;
add.f64 %fd47, %fd46, %fd42;
st.global.f64 [%rd3+64000], %fd47;
ld.global.f64 %fd48, [%rd4];
fma.rn.f64 %fd49, %fd41, %fd1, %fd48;
ld.global.f64 %fd50, [%rd4+64000];
st.global.f64 [%rd4], %fd49;
ld.global.f64 %fd51, [%rd3+64000];
fma.rn.f64 %fd52, %fd51, %fd1, %fd50;
st.global.f64 [%rd4+64000], %fd52;
ld.global.f64 %fd53, [%rd1];
mul.f64 %fd54, %fd53, %fd1;
ld.global.f64 %fd55, [%rd2];
div.rn.f64 %fd56, %fd54, %fd55;
ld.global.f64 %fd57, [%rd3];
add.f64 %fd58, %fd56, %fd57;
ld.global.f64 %fd59, [%rd3+64000];
st.global.f64 [%rd3], %fd58;
ld.global.f64 %fd60, [%rd1+64000];
mul.f64 %fd61, %fd60, %fd1;
ld.global.f64 %fd62, [%rd2];
div.rn.f64 %fd63, %fd61, %fd62;
add.f64 %fd64, %fd63, %fd59;
st.global.f64 [%rd3+64000], %fd64;
ld.global.f64 %fd65, [%rd4];
fma.rn.f64 %fd66, %fd58, %fd1, %fd65;
ld.global.f64 %fd67, [%rd4+64000];
st.global.f64 [%rd4], %fd66;
ld.global.f64 %fd68, [%rd3+64000];
fma.rn.f64 %fd69, %fd68, %fd1, %fd67;
st.global.f64 [%rd4+64000], %fd69;
ld.global.f64 %fd70, [%rd1];
mul.f64 %fd71, %fd70, %fd1;
ld.global.f64 %fd72, [%rd2];
div.rn.f64 %fd73, %fd71, %fd72;
ld.global.f64 %fd74, [%rd3];
add.f64 %fd75, %fd73, %fd74;
ld.global.f64 %fd76, [%rd3+64000];
st.global.f64 [%rd3], %fd75;
ld.global.f64 %fd77, [%rd1+64000];
mul.f64 %fd78, %fd77, %fd1;
ld.global.f64 %fd79, [%rd2];
div.rn.f64 %fd80, %fd78, %fd79;
add.f64 %fd81, %fd80, %fd76;
st.global.f64 [%rd3+64000], %fd81;
ld.global.f64 %fd82, [%rd4];
fma.rn.f64 %fd83, %fd75, %fd1, %fd82;
ld.global.f64 %fd84, [%rd4+64000];
st.global.f64 [%rd4], %fd83;
ld.global.f64 %fd85, [%rd3+64000];
fma.rn.f64 %fd86, %fd85, %fd1, %fd84;
st.global.f64 [%rd4+64000], %fd86;
add.s32 %r4, %r4, 5;
setp.ne.s32	%p1, %r4, 0;
@%p1 bra BB6_1;

ret;
}


.visible .entry _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_11reset_forceEvEEE4callEPS4_mEUlS8_mE_JS8_mEEEvT_DpT0_(
.param .align 1 .b8 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_11reset_forceEvEEE4callEPS4_mEUlS8_mE_JS8_mEEEvT_DpT0__param_0[1],
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_11reset_forceEvEEE4callEPS4_mEUlS8_mE_JS8_mEEEvT_DpT0__param_1,
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_11reset_forceEvEEE4callEPS4_mEUlS8_mE_JS8_mEEEvT_DpT0__param_2
)
{
.reg .pred %p<2>;
.reg .b32 %r<5>;
.reg .b64 %rd<9>;


ld.param.u64 %rd2, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_11reset_forceEvEEE4callEPS4_mEUlS8_mE_JS8_mEEEvT_DpT0__param_1];
ld.param.u64 %rd3, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_11reset_forceEvEEE4callEPS4_mEUlS8_mE_JS8_mEEEvT_DpT0__param_2];
mov.u32 %r1, %tid.x;
mov.u32 %r2, %ntid.x;
mov.u32 %r3, %ctaid.x;
mad.lo.s32 %r4, %r2, %r3, %r1;
cvt.s64.s32	%rd1, %r4;
setp.ge.u64	%p1, %rd1, %rd3;
@%p1 bra BB7_2;

add.s64 %rd4, %rd1, %rd2;
shl.b64 %rd5, %rd4, 3;
mov.u64 %rd6, __Bodydata_buffer;
add.s64 %rd7, %rd5, %rd6;
mov.u64 %rd8, 0;
st.global.u64 [%rd7+320016], %rd8;
st.global.u64 [%rd7+384016], %rd8;

BB7_2:
ret;
}


.visible .entry _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_22add_all_forces_to_thisEvEEE4callEPS4_mEUlS8_mE_JS8_mEEEvT_DpT0_(
.param .align 1 .b8 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_22add_all_forces_to_thisEvEEE4callEPS4_mEUlS8_mE_JS8_mEEEvT_DpT0__param_0[1],
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_22add_all_forces_to_thisEvEEE4callEPS4_mEUlS8_mE_JS8_mEEEvT_DpT0__param_1,
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_22add_all_forces_to_thisEvEEE4callEPS4_mEUlS8_mE_JS8_mEEEvT_DpT0__param_2
)
{
.reg .pred %p<7>;
.reg .b32 %r<21>;
.reg .f64 %fd<93>;
.reg .b64 %rd<61>;


ld.param.u64 %rd15, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_22add_all_forces_to_thisEvEEE4callEPS4_mEUlS8_mE_JS8_mEEEvT_DpT0__param_1];
ld.param.u64 %rd16, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_22add_all_forces_to_thisEvEEE4callEPS4_mEUlS8_mE_JS8_mEEEvT_DpT0__param_2];
mov.u32 %r1, %tid.x;
mov.u32 %r2, %ntid.x;
mov.u32 %r3, %ctaid.x;
mad.lo.s32 %r4, %r2, %r3, %r1;
cvt.s64.s32	%rd1, %r4;
setp.ge.u64	%p1, %rd1, %rd16;
@%p1 bra BB8_11;

add.s64 %rd2, %rd1, %rd15;
mov.u64 %rd60, 8000;
mov.u64 %rd59, __Bodydata_buffer;
mov.u64 %rd58, 4;
mov.u64 %rd57, 3;
mov.u64 %rd56, 2;
mov.u64 %rd55, 1;

BB8_2:
setp.eq.s64	%p2, %rd55, %rd2;
@%p2 bra BB8_4;

shl.b64 %rd23, %rd55, 3;
mov.u64 %rd24, __Bodydata_buffer;
add.s64 %rd25, %rd24, 16;
add.s64 %rd26, %rd23, %rd25;
shl.b64 %rd29, %rd2, 3;
add.s64 %rd30, %rd25, %rd29;
ld.global.f64 %fd1, [%rd30+64000];
ld.global.f64 %fd2, [%rd26+64000];
sub.f64 %fd3, %fd2, %fd1;
ld.global.f64 %fd4, [%rd30+128000];
ld.global.f64 %fd5, [%rd26+128000];
sub.f64 %fd6, %fd5, %fd4;
mul.f64 %fd7, %fd6, %fd6;
fma.rn.f64 %fd8, %fd3, %fd3, %fd7;
sqrt.rn.f64 %fd9, %fd8;
ld.global.f64 %fd10, [%rd30];
mul.f64 %fd11, %fd10, 0d3DD257B4D002790D;
ld.global.f64 %fd12, [%rd59+24];
mul.f64 %fd13, %fd11, %fd12;
fma.rn.f64 %fd14, %fd9, %fd9, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd15, %fd13, %fd14;
mul.f64 %fd16, %fd3, %fd15;
div.rn.f64 %fd17, %fd16, %fd9;
ld.global.f64 %fd18, [%rd30+320000];
add.f64 %fd19, %fd17, %fd18;
ld.global.f64 %fd20, [%rd30+384000];
st.global.f64 [%rd30+320000], %fd19;
mul.f64 %fd21, %fd6, %fd15;
div.rn.f64 %fd22, %fd21, %fd9;
add.f64 %fd23, %fd22, %fd20;
st.global.f64 [%rd30+384000], %fd23;

BB8_4:
setp.eq.s64	%p3, %rd56, %rd2;
@%p3 bra BB8_6;

shl.b64 %rd31, %rd56, 3;
mov.u64 %rd32, __Bodydata_buffer;
add.s64 %rd33, %rd32, 16;
add.s64 %rd34, %rd31, %rd33;
shl.b64 %rd37, %rd2, 3;
add.s64 %rd38, %rd33, %rd37;
ld.global.f64 %fd24, [%rd38+64000];
ld.global.f64 %fd25, [%rd34+64000];
sub.f64 %fd26, %fd25, %fd24;
ld.global.f64 %fd27, [%rd38+128000];
ld.global.f64 %fd28, [%rd34+128000];
sub.f64 %fd29, %fd28, %fd27;
mul.f64 %fd30, %fd29, %fd29;
fma.rn.f64 %fd31, %fd26, %fd26, %fd30;
sqrt.rn.f64 %fd32, %fd31;
ld.global.f64 %fd33, [%rd38];
mul.f64 %fd34, %fd33, 0d3DD257B4D002790D;
ld.global.f64 %fd35, [%rd59+32];
mul.f64 %fd36, %fd34, %fd35;
fma.rn.f64 %fd37, %fd32, %fd32, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd38, %fd36, %fd37;
mul.f64 %fd39, %fd26, %fd38;
div.rn.f64 %fd40, %fd39, %fd32;
ld.global.f64 %fd41, [%rd38+320000];
add.f64 %fd42, %fd40, %fd41;
ld.global.f64 %fd43, [%rd38+384000];
st.global.f64 [%rd38+320000], %fd42;
mul.f64 %fd44, %fd29, %fd38;
div.rn.f64 %fd45, %fd44, %fd32;
add.f64 %fd46, %fd45, %fd43;
st.global.f64 [%rd38+384000], %fd46;

BB8_6:
setp.eq.s64	%p4, %rd57, %rd2;
@%p4 bra BB8_8;

shl.b64 %rd39, %rd57, 3;
mov.u64 %rd40, __Bodydata_buffer;
add.s64 %rd41, %rd40, 16;
add.s64 %rd42, %rd39, %rd41;
shl.b64 %rd45, %rd2, 3;
add.s64 %rd46, %rd41, %rd45;
ld.global.f64 %fd47, [%rd46+64000];
ld.global.f64 %fd48, [%rd42+64000];
sub.f64 %fd49, %fd48, %fd47;
ld.global.f64 %fd50, [%rd46+128000];
ld.global.f64 %fd51, [%rd42+128000];
sub.f64 %fd52, %fd51, %fd50;
mul.f64 %fd53, %fd52, %fd52;
fma.rn.f64 %fd54, %fd49, %fd49, %fd53;
sqrt.rn.f64 %fd55, %fd54;
ld.global.f64 %fd56, [%rd46];
mul.f64 %fd57, %fd56, 0d3DD257B4D002790D;
ld.global.f64 %fd58, [%rd59+40];
mul.f64 %fd59, %fd57, %fd58;
fma.rn.f64 %fd60, %fd55, %fd55, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd61, %fd59, %fd60;
mul.f64 %fd62, %fd49, %fd61;
div.rn.f64 %fd63, %fd62, %fd55;
ld.global.f64 %fd64, [%rd46+320000];
add.f64 %fd65, %fd63, %fd64;
ld.global.f64 %fd66, [%rd46+384000];
st.global.f64 [%rd46+320000], %fd65;
mul.f64 %fd67, %fd52, %fd61;
div.rn.f64 %fd68, %fd67, %fd55;
add.f64 %fd69, %fd68, %fd66;
st.global.f64 [%rd46+384000], %fd69;

BB8_8:
setp.eq.s64	%p5, %rd58, %rd2;
@%p5 bra BB8_10;

shl.b64 %rd47, %rd58, 3;
mov.u64 %rd48, __Bodydata_buffer;
add.s64 %rd49, %rd48, 16;
add.s64 %rd50, %rd47, %rd49;
shl.b64 %rd53, %rd2, 3;
add.s64 %rd54, %rd49, %rd53;
ld.global.f64 %fd70, [%rd54+64000];
ld.global.f64 %fd71, [%rd50+64000];
sub.f64 %fd72, %fd71, %fd70;
ld.global.f64 %fd73, [%rd54+128000];
ld.global.f64 %fd74, [%rd50+128000];
sub.f64 %fd75, %fd74, %fd73;
mul.f64 %fd76, %fd75, %fd75;
fma.rn.f64 %fd77, %fd72, %fd72, %fd76;
sqrt.rn.f64 %fd78, %fd77;
ld.global.f64 %fd79, [%rd54];
mul.f64 %fd80, %fd79, 0d3DD257B4D002790D;
ld.global.f64 %fd81, [%rd59+48];
mul.f64 %fd82, %fd80, %fd81;
fma.rn.f64 %fd83, %fd78, %fd78, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd84, %fd82, %fd83;
mul.f64 %fd85, %fd72, %fd84;
div.rn.f64 %fd86, %fd85, %fd78;
ld.global.f64 %fd87, [%rd54+320000];
add.f64 %fd88, %fd86, %fd87;
ld.global.f64 %fd89, [%rd54+384000];
st.global.f64 [%rd54+320000], %fd88;
mul.f64 %fd90, %fd75, %fd84;
div.rn.f64 %fd91, %fd90, %fd78;
add.f64 %fd92, %fd91, %fd89;
st.global.f64 [%rd54+384000], %fd92;

BB8_10:
add.s64 %rd60, %rd60, -4;
add.s64 %rd59, %rd59, 32;
add.s64 %rd58, %rd58, 4;
add.s64 %rd57, %rd57, 4;
add.s64 %rd56, %rd56, 4;
add.s64 %rd55, %rd55, 4;
setp.ne.s64	%p6, %rd60, 0;
@%p6 bra BB8_2;

BB8_11:
ret;
}


.visible .entry _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0_(
.param .align 1 .b8 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0__param_0[1],
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0__param_1,
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0__param_2,
.param .f64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0__param_3
)
{
.reg .pred %p<8>;
.reg .b32 %r<5>;
.reg .f64 %fd<25>;
.reg .b64 %rd<10>;


ld.param.u64 %rd4, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0__param_1];
ld.param.u64 %rd5, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0__param_2];
ld.param.f64 %fd4, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0__param_3];
mov.u32 %r1, %tid.x;
mov.u32 %r2, %ntid.x;
mov.u32 %r3, %ctaid.x;
mad.lo.s32 %r4, %r2, %r3, %r1;
cvt.s64.s32	%rd1, %r4;
setp.ge.u64	%p1, %rd1, %rd5;
@%p1 bra BB9_5;

add.s64 %rd6, %rd1, %rd4;
shl.b64 %rd7, %rd6, 3;
mov.u64 %rd8, __Bodydata_buffer;
add.s64 %rd9, %rd8, %rd7;
ld.global.f64 %fd5, [%rd9+320016];
mul.f64 %fd6, %fd5, %fd4;
ld.global.f64 %fd7, [%rd9+16];
div.rn.f64 %fd8, %fd6, %fd7;
add.s64 %rd2, %rd9, 192016;
ld.global.f64 %fd9, [%rd9+192016];
add.f64 %fd10, %fd8, %fd9;
ld.global.f64 %fd11, [%rd9+384016];
ld.global.f64 %fd12, [%rd9+256016];
ld.global.f64 %fd13, [%rd9+64016];
ld.global.f64 %fd14, [%rd9+128016];
st.global.f64 [%rd9+192016], %fd10;
mul.f64 %fd15, %fd11, %fd4;
fma.rn.f64 %fd16, %fd10, %fd4, %fd13;
st.global.f64 [%rd9+64016], %fd16;
ld.global.f64 %fd17, [%rd9+16];
div.rn.f64 %fd18, %fd15, %fd17;
add.f64 %fd19, %fd18, %fd12;
st.global.f64 [%rd9+256016], %fd19;
add.s64 %rd3, %rd9, 64016;
fma.rn.f64 %fd24, %fd19, %fd4, %fd14;
st.global.f64 [%rd9+128016], %fd24;
setp.lt.f64	%p2, %fd16, 0dBFF0000000000000;
setp.gt.f64	%p3, %fd16, 0d3FF0000000000000;
or.pred %p4, %p2, %p3;
@!%p4 bra BB9_3;
bra.uni BB9_2;

BB9_2:
ld.global.f64 %fd20, [%rd2];
neg.f64 %fd21, %fd20;
st.global.f64 [%rd2], %fd21;
ld.global.f64 %fd24, [%rd3+64000];

BB9_3:
setp.gt.f64	%p5, %fd24, 0d3FF0000000000000;
setp.lt.f64	%p6, %fd24, 0dBFF0000000000000;
or.pred %p7, %p6, %p5;
@!%p7 bra BB9_5;
bra.uni BB9_4;

BB9_4:
ld.global.f64 %fd22, [%rd2+64000];
neg.f64 %fd23, %fd22;
st.global.f64 [%rd2+64000], %fd23;

BB9_5:
ret;
}


.visible .entry _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0_(
.param .align 1 .b8 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0__param_0[1],
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0__param_1,
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0__param_2,
.param .f64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0__param_3
)
{
.reg .pred %p<3>;
.reg .b32 %r<9>;
.reg .f64 %fd<87>;
.reg .b64 %rd<12>;


ld.param.u64 %rd6, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0__param_1];
ld.param.u64 %rd7, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0__param_2];
ld.param.f64 %fd1, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEPS4_mdEUlS8_mdE_JS8_mdEEEvT_DpT0__param_3];
mov.u32 %r3, %tid.x;
mov.u32 %r4, %ntid.x;
mov.u32 %r5, %ctaid.x;
mad.lo.s32 %r6, %r4, %r5, %r3;
cvt.s64.s32	%rd1, %r6;
setp.ge.u64	%p1, %rd1, %rd7;
@%p1 bra BB10_3;

add.s64 %rd8, %rd1, %rd6;
shl.b64 %rd9, %rd8, 3;
mov.u64 %rd10, __Bodydata_buffer;
add.s64 %rd11, %rd10, %rd9;
add.s64 %rd3, %rd11, 16;
add.s64 %rd2, %rd11, 320016;
add.s64 %rd4, %rd11, 192016;
add.s64 %rd5, %rd11, 64016;
mov.u32 %r8, -1000;

BB10_2:
ld.global.f64 %fd2, [%rd2];
mul.f64 %fd3, %fd2, %fd1;
ld.global.f64 %fd4, [%rd3];
div.rn.f64 %fd5, %fd3, %fd4;
ld.global.f64 %fd6, [%rd4];
add.f64 %fd7, %fd5, %fd6;
ld.global.f64 %fd8, [%rd4+64000];
st.global.f64 [%rd4], %fd7;
ld.global.f64 %fd9, [%rd2+64000];
mul.f64 %fd10, %fd9, %fd1;
ld.global.f64 %fd11, [%rd3];
div.rn.f64 %fd12, %fd10, %fd11;
add.f64 %fd13, %fd12, %fd8;
st.global.f64 [%rd4+64000], %fd13;
ld.global.f64 %fd14, [%rd5];
fma.rn.f64 %fd15, %fd7, %fd1, %fd14;
ld.global.f64 %fd16, [%rd5+64000];
st.global.f64 [%rd5], %fd15;
ld.global.f64 %fd17, [%rd4+64000];
fma.rn.f64 %fd18, %fd17, %fd1, %fd16;
st.global.f64 [%rd5+64000], %fd18;
ld.global.f64 %fd19, [%rd2];
mul.f64 %fd20, %fd19, %fd1;
ld.global.f64 %fd21, [%rd3];
div.rn.f64 %fd22, %fd20, %fd21;
ld.global.f64 %fd23, [%rd4];
add.f64 %fd24, %fd22, %fd23;
ld.global.f64 %fd25, [%rd4+64000];
st.global.f64 [%rd4], %fd24;
ld.global.f64 %fd26, [%rd2+64000];
mul.f64 %fd27, %fd26, %fd1;
ld.global.f64 %fd28, [%rd3];
div.rn.f64 %fd29, %fd27, %fd28;
add.f64 %fd30, %fd29, %fd25;
st.global.f64 [%rd4+64000], %fd30;
ld.global.f64 %fd31, [%rd5];
fma.rn.f64 %fd32, %fd24, %fd1, %fd31;
ld.global.f64 %fd33, [%rd5+64000];
st.global.f64 [%rd5], %fd32;
ld.global.f64 %fd34, [%rd4+64000];
fma.rn.f64 %fd35, %fd34, %fd1, %fd33;
st.global.f64 [%rd5+64000], %fd35;
ld.global.f64 %fd36, [%rd2];
mul.f64 %fd37, %fd36, %fd1;
ld.global.f64 %fd38, [%rd3];
div.rn.f64 %fd39, %fd37, %fd38;
ld.global.f64 %fd40, [%rd4];
add.f64 %fd41, %fd39, %fd40;
ld.global.f64 %fd42, [%rd4+64000];
st.global.f64 [%rd4], %fd41;
ld.global.f64 %fd43, [%rd2+64000];
mul.f64 %fd44, %fd43, %fd1;
ld.global.f64 %fd45, [%rd3];
div.rn.f64 %fd46, %fd44, %fd45;
add.f64 %fd47, %fd46, %fd42;
st.global.f64 [%rd4+64000], %fd47;
ld.global.f64 %fd48, [%rd5];
fma.rn.f64 %fd49, %fd41, %fd1, %fd48;
ld.global.f64 %fd50, [%rd5+64000];
st.global.f64 [%rd5], %fd49;
ld.global.f64 %fd51, [%rd4+64000];
fma.rn.f64 %fd52, %fd51, %fd1, %fd50;
st.global.f64 [%rd5+64000], %fd52;
ld.global.f64 %fd53, [%rd2];
mul.f64 %fd54, %fd53, %fd1;
ld.global.f64 %fd55, [%rd3];
div.rn.f64 %fd56, %fd54, %fd55;
ld.global.f64 %fd57, [%rd4];
add.f64 %fd58, %fd56, %fd57;
ld.global.f64 %fd59, [%rd4+64000];
st.global.f64 [%rd4], %fd58;
ld.global.f64 %fd60, [%rd2+64000];
mul.f64 %fd61, %fd60, %fd1;
ld.global.f64 %fd62, [%rd3];
div.rn.f64 %fd63, %fd61, %fd62;
add.f64 %fd64, %fd63, %fd59;
st.global.f64 [%rd4+64000], %fd64;
ld.global.f64 %fd65, [%rd5];
fma.rn.f64 %fd66, %fd58, %fd1, %fd65;
ld.global.f64 %fd67, [%rd5+64000];
st.global.f64 [%rd5], %fd66;
ld.global.f64 %fd68, [%rd4+64000];
fma.rn.f64 %fd69, %fd68, %fd1, %fd67;
st.global.f64 [%rd5+64000], %fd69;
ld.global.f64 %fd70, [%rd2];
mul.f64 %fd71, %fd70, %fd1;
ld.global.f64 %fd72, [%rd3];
div.rn.f64 %fd73, %fd71, %fd72;
ld.global.f64 %fd74, [%rd4];
add.f64 %fd75, %fd73, %fd74;
ld.global.f64 %fd76, [%rd4+64000];
st.global.f64 [%rd4], %fd75;
ld.global.f64 %fd77, [%rd2+64000];
mul.f64 %fd78, %fd77, %fd1;
ld.global.f64 %fd79, [%rd3];
div.rn.f64 %fd80, %fd78, %fd79;
add.f64 %fd81, %fd80, %fd76;
st.global.f64 [%rd4+64000], %fd81;
ld.global.f64 %fd82, [%rd5];
fma.rn.f64 %fd83, %fd75, %fd1, %fd82;
ld.global.f64 %fd84, [%rd5+64000];
st.global.f64 [%rd5], %fd83;
ld.global.f64 %fd85, [%rd4+64000];
fma.rn.f64 %fd86, %fd85, %fd1, %fd84;
st.global.f64 [%rd5+64000], %fd86;
add.s32 %r8, %r8, 5;
setp.ne.s32	%p2, %r8, 0;
@%p2 bra BB10_2;

BB10_3:
ret;
}


.visible .entry _ZN4ikra3soa28storage_increase_size_kernelEPNS0_19StorageStrategyBaseEm(
.param .u64 _ZN4ikra3soa28storage_increase_size_kernelEPNS0_19StorageStrategyBaseEm_param_0,
.param .u64 _ZN4ikra3soa28storage_increase_size_kernelEPNS0_19StorageStrategyBaseEm_param_1
)
{
.reg .b64 %rd<6>;


ld.param.u64 %rd1, [_ZN4ikra3soa28storage_increase_size_kernelEPNS0_19StorageStrategyBaseEm_param_0];
ld.param.u64 %rd2, [_ZN4ikra3soa28storage_increase_size_kernelEPNS0_19StorageStrategyBaseEm_param_1];
cvta.to.global.u64 %rd3, %rd1;
ld.global.u64 %rd4, [%rd3];
add.s64 %rd5, %rd4, %rd2;
st.global.u64 [%rd3], %rd5;
st.global.u64 [_ZN4ikra3soa23d_previous_storage_sizeE], %rd4;
ret;
}


.visible .entry _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I4BodyLm0EEEJEEEvPT_DpT0_(
.param .u64 _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I4BodyLm0EEEJEEEvPT_DpT0__param_0
)
{
.reg .pred %p<2>;
.reg .b64 %rd<4>;


ld.param.u64 %rd1, [_ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I4BodyLm0EEEJEEEvPT_DpT0__param_0];
setp.eq.s64	%p1, %rd1, 0;
@%p1 bra BB12_2;

cvta.to.global.u64 %rd2, %rd1;
mov.u64 %rd3, 0;
st.global.u64 [%rd2], %rd3;
st.global.u64 [%rd2+8], %rd3;

BB12_2:
ret;
}


