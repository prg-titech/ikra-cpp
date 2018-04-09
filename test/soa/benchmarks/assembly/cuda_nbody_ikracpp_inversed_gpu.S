
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
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEdEUldE_JdEEEvT_DpT0_:
  REG:34 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:32 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEdEUldE_JdEEEvT_DpT0_:
  REG:27 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:32 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_22add_all_forces_to_thisEvEEE4callEvEUlvE_JEEEvT_DpT0_:
  REG:46 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:321 CONSTANT[2]:60 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_11reset_forceEvEEE4callEvEUlvE_JEEEvT_DpT0_:
  REG:8 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:321 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z22codengen_simple_updateP4Bodyd:
  REG:34 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:32 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z18codegen_reset_massP4Body:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z24codegen_reset_force_halfP4Body:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z26codegen_reset_force_manualP4Body:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z14codegen_updateP4Bodyd:
  REG:27 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:32 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z17codegen_add_forceP4BodyS0_:
  REG:32 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:60 TEXTURE:0 SURFACE:0 SAMPLER:0
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


		Function : _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEdEUldE_JdEEEvT_DpT0_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                              /* 0x22804280423042b7 */
        /*0008*/                   MOV R1, c[0x0][0x44];                                      /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R7, 0x0;                                            /* 0x180000000001dde2 */
        /*0018*/                   S2R R4, SR_TID.X;                                          /* 0x2c00000084011c04 */
        /*0020*/                   MOV32I R6, 0x0;                                            /* 0x1800000000019de2 */
        /*0028*/                   S2R R0, SR_CTAID.X;                                        /* 0x2c00000094001c04 */
        /*0030*/                   MOV R2, R7;                                                /* 0x280000001c009de4 */
        /*0038*/                   MOV R3, R6;                                                /* 0x280000001800dde4 */
                                                                                              /* 0x2282e2c2c3f28047 */
        /*0048*/                   LD.E.64 R2, [R2];                                          /* 0x8400000000209ca5 */
        /*0050*/                   IMAD R4, R0, c[0x0][0x28], R4;                             /* 0x20084000a0011ca3 */
        /*0058*/                   SHR R5, R4, 0x1f;                                          /* 0x5800c0007c415c23 */
        /*0060*/                   ISUB RZ.CC, R4, R2;                                        /* 0x48010000084fdd03 */
        /*0068*/                   ISETP.GE.U32.X.AND P0, PT, R5, R3, PT;                     /* 0x1b0e00000c51dc43 */
        /*0070*/               @P0 EXIT;                                                      /* 0x80000000000001e7 */
        /*0078*/                   MOV32I R0, 0x3;                                            /* 0x180000000c001de2 */
                                                                                              /* 0x2202b20242323047 */
        /*0088*/                   IADD32I RZ.CC, R0, -0x20;                                  /* 0x0fffffff800fdc02 */
        /*0090*/                   SHL R5, R5, 0x3;                                           /* 0x6000c0000c515c03 */
        /*0098*/                   SHL R0, R4, -0x1d;                                         /* 0x6000ffff8c401c03 */
        /*00a0*/                   SHR.U32 R2, R4, 0x1d;                                      /* 0x5800c00074409c03 */
        /*00a8*/                   CSETP.SFT.AND P0, PT, CC, PT;                              /* 0x040e00000151dc04 */
        /*00b0*/                   IMAD.U32.U32 R4.CC, R4, 0x8, R7;                           /* 0x200fc00020411c03 */
        /*00b8*/               @P0 IADD R0, R5, R2;                                           /* 0x4800000008500003 */
                                                                                              /* 0x23f0423042b04277 */
        /*00c8*/                   MOV R2, R4;                                                /* 0x2800000010009de4 */
        /*00d0*/                   IADD.X R3, R0, R6;                                         /* 0x480000001800dc43 */
        /*00d8*/                   MOV32I R0, 0xfffffc18;                                     /* 0x1bfffff060001de2 */
        /*00e0*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
        /*00e8*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*00f0*/                   LD.E.64 R4, [R2+0x4e218];                                  /* 0x8400138860211ca5 */
        /*00f8*/                   SSY 0x198;                                                 /* 0x6000000260001c07 */
                                                                                              /* 0x2292929252329287 */
        /*0108*/                   MUFU.RCP64H R11, R7;                                       /* 0xc80000001872dc00 */
        /*0110*/                   DFMA R8, -R6, R10, c[0x2][0x0];                            /* 0x2014880000621e01 */
        /*0118*/                   DFMA R12, R8, R8, R8;                                      /* 0x2010000020831c01 */
        /*0120*/                   DMUL R8, R4, c[0x0][0x148];                                /* 0x5000400520421c01 */
        /*0128*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*0130*/                   DMUL R4, R8, R10;                                          /* 0x5000000028811c01 */
        /*0138*/                   DFMA R14, -R6, R4, R8;                                     /* 0x2010000010639e01 */
                                                                                              /* 0x22b042b2c2828207 */
        /*0148*/                   DFMA R10, R10, R14, R4;                                    /* 0x2008000038a29c01 */
        /*0150*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*0158*/                   FFMA R4, RZ, R7, R11;                                      /* 0x301600001ff11c00 */
        /*0160*/                   FSETP.GT.AND P0, PT, |R4|, c[0x2][0xc], PT;                /* 0x220e48003041dc80 */
        /*0168*/               @P0 BRA CC.NEU, 0x190;                                         /* 0x40000000800001a7 */
        /*0170*/                   MOV R4, R6;                                                /* 0x2800000018011de4 */
        /*0178*/                   MOV R5, R7;                                                /* 0x280000001c015de4 */
                                                                                              /* 0x223230423042b2b7 */
        /*0188*/                   CAL 0xa78;                                                 /* 0x50000023a0011c07 */
        /*0190*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0198*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
        /*01a0*/                   MOV32I R8, 0x1;                                            /* 0x1800000004021de2 */
        /*01a8*/                   LD.E.64 R6, [R2+0x5dc18];                                  /* 0x8400177060219ca5 */
        /*01b0*/                   SSY 0x270;                                                 /* 0x60000002e0001c07 */
        /*01b8*/                   LD.E.64 R24, [R2+0x2ee18];                                 /* 0x84000bb860261ca5 */
                                                                                              /* 0x22929232929283f7 */
        /*01c8*/                   LD.E.64 R22, [R2+0x3e818];                                 /* 0x84000fa060259ca5 */
        /*01d0*/                   MUFU.RCP64H R9, R5;                                        /* 0xc800000018525c00 */
        /*01d8*/                   DFMA R12, -R4, R8, c[0x2][0x0];                            /* 0x2010880000431e01 */
        /*01e0*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*01e8*/                   DFMA R12, R8, R12, R8;                                     /* 0x2010000030831c01 */
        /*01f0*/                   DMUL R8, R6, c[0x0][0x148];                                /* 0x5000400520621c01 */
        /*01f8*/                   DMUL R6, R8, R12;                                          /* 0x5000000030819c01 */
                                                                                              /* 0x22c2720042923297 */
        /*0208*/                   DFMA R16, -R4, R6, R8;                                     /* 0x2010000018441e01 */
        /*0210*/                   DFMA R16, R12, R16, R6;                                    /* 0x200c000040c41c01 */
        /*0218*/                   DADD R24, R10, R24;                                        /* 0x4800000060a61c01 */
        /*0220*/                   FFMA R6, RZ, R5, R17;                                      /* 0x3022000017f19c00 */
        /*0228*/                   ST.E.64 [R2+0x2ee18], R24;                                 /* 0x94000bb860261ca5 */
        /*0230*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*0238*/                   FSETP.GT.AND P0, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003061dc80 */
                                                                                              /* 0x223042b2b042b2b7 */
        /*0248*/               @P0 BRA CC.NEU, 0x268;                                         /* 0x40000000600001a7 */
        /*0250*/                   CAL 0xa78;                                                 /* 0x5000002080011c07 */
        /*0258*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
        /*0260*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*0268*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0270*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
        /*0278*/                   DADD R16, R22, R16;                                        /* 0x4800000041641c01 */
                                                                                              /* 0x23f2e2e230423047 */
        /*0288*/                   LD.E.64 R12, [R2+0x4e218];                                 /* 0x8400138860231ca5 */
        /*0290*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0298*/                   LD.E.64 R6, [R2+0xfa18];                                   /* 0x840003e860219ca5 */
        /*02a0*/                   SSY 0x390;                                                 /* 0x60000003a0001c07 */
        /*02a8*/                   LD.E.64 R8, [R2+0x1f418];                                  /* 0x840007d060221ca5 */
        /*02b0*/                   ST.E.64 [R2+0x3e818], R16;                                 /* 0x94000fa060241ca5 */
        /*02b8*/                   LD.E.64 R10, [R2+0x3e818];                                 /* 0x84000fa060229ca5 */
                                                                                              /* 0x2292929232929287 */
        /*02c8*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*02d0*/                   DFMA R18, -R4, R14, c[0x2][0x0];                           /* 0x201c880000449e01 */
        /*02d8*/                   DFMA R18, R18, R18, R18;                                   /* 0x2024000049249c01 */
        /*02e0*/                   DFMA R18, R14, R18, R14;                                   /* 0x201c000048e49c01 */
        /*02e8*/                   DMUL R14, R12, c[0x0][0x148];                              /* 0x5000400520c39c01 */
        /*02f0*/                   DMUL R32, R14, R18;                                        /* 0x5000000048e81c01 */
        /*02f8*/                   DFMA R12, -R4, R32, R14;                                   /* 0x201c000080431e01 */
                                                                                              /* 0x22c2720042523237 */
        /*0308*/                   DFMA R18, R18, R12, R32;                                   /* 0x2040000031249c01 */
        /*0310*/                   DFMA R6, R24, c[0x0][0x148], R6;                           /* 0x200c400521819c01 */
        /*0318*/                   DFMA R8, R10, c[0x0][0x148], R8;                           /* 0x2010400520a21c01 */
        /*0320*/                   FFMA R10, RZ, R5, R19;                                     /* 0x3026000017f29c00 */
        /*0328*/                   ST.E.64 [R2+0xfa18], R6;                                   /* 0x940003e860219ca5 */
        /*0330*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
        /*0338*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
                                                                                              /* 0x22b042b2b042b047 */
        /*0348*/                   ST.E.64 [R2+0x1f418], R8;                                  /* 0x940007d060221ca5 */
        /*0350*/               @P0 BRA CC.NEU, 0x380;                                         /* 0x40000000a00001a7 */
        /*0358*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0360*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
        /*0368*/                   CAL 0xa78;                                                 /* 0x5000001c20011c07 */
        /*0370*/                   MOV R18, R10;                                              /* 0x2800000028049de4 */
        /*0378*/                   MOV R19, R11;                                              /* 0x280000002c04dde4 */
                                                                                              /* 0x23f23230423042b7 */
        /*0388*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0390*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
        /*0398*/                   MOV32I R8, 0x1;                                            /* 0x1800000004021de2 */
        /*03a0*/                   LD.E.64 R6, [R2+0x5dc18];                                  /* 0x8400177060219ca5 */
        /*03a8*/                   SSY 0x458;                                                 /* 0x60000002a0001c07 */
        /*03b0*/                   LD.E.64 R24, [R2+0x2ee18];                                 /* 0x84000bb860261ca5 */
        /*03b8*/                   LD.E.64 R22, [R2+0x3e818];                                 /* 0x84000fa060259ca5 */
                                                                                              /* 0x2292929232929287 */
        /*03c8*/                   MUFU.RCP64H R9, R5;                                        /* 0xc800000018525c00 */
        /*03d0*/                   DFMA R10, -R4, R8, c[0x2][0x0];                            /* 0x2010880000429e01 */
        /*03d8*/                   DFMA R10, R10, R10, R10;                                   /* 0x2014000028a29c01 */
        /*03e0*/                   DFMA R10, R8, R10, R8;                                     /* 0x2010000028829c01 */
        /*03e8*/                   DMUL R8, R6, c[0x0][0x148];                                /* 0x5000400520621c01 */
        /*03f0*/                   DMUL R6, R8, R10;                                          /* 0x5000000028819c01 */
        /*03f8*/                   DFMA R12, -R4, R6, R8;                                     /* 0x2010000018431e01 */
                                                                                              /* 0x22b2c27200429237 */
        /*0408*/                   DFMA R10, R10, R12, R6;                                    /* 0x200c000030a29c01 */
        /*0410*/                   DADD R24, R18, R24;                                        /* 0x4800000061261c01 */
        /*0418*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*0420*/                   ST.E.64 [R2+0x2ee18], R24;                                 /* 0x94000bb860261ca5 */
        /*0428*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*0430*/                   FSETP.GT.AND P0, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003061dc80 */
        /*0438*/               @P0 BRA CC.NEU, 0x450;                                         /* 0x40000000400001a7 */
                                                                                              /* 0x223230423042b2b7 */
        /*0448*/                   CAL 0xa78;                                                 /* 0x50000018a0011c07 */
        /*0450*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0458*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
        /*0460*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*0468*/                   LD.E.64 R12, [R2+0x4e218];                                 /* 0x8400138860231ca5 */
        /*0470*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0478*/                   LD.E.64 R6, [R2+0xfa18];                                   /* 0x840003e860219ca5 */
                                                                                              /* 0x22929283f2e2e047 */
        /*0488*/                   LD.E.64 R8, [R2+0x1f418];                                  /* 0x840007d060221ca5 */
        /*0490*/                   SSY 0x570;                                                 /* 0x6000000360001c07 */
        /*0498*/                   ST.E.64 [R2+0x3e818], R22;                                 /* 0x94000fa060259ca5 */
        /*04a0*/                   LD.E.64 R10, [R2+0x3e818];                                 /* 0x84000fa060229ca5 */
        /*04a8*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*04b0*/                   DFMA R16, -R4, R14, c[0x2][0x0];                           /* 0x201c880000441e01 */
        /*04b8*/                   DFMA R16, R16, R16, R16;                                   /* 0x2020000041041c01 */
                                                                                              /* 0x2252323292929237 */
        /*04c8*/                   DFMA R16, R14, R16, R14;                                   /* 0x201c000040e41c01 */
        /*04d0*/                   DMUL R14, R12, c[0x0][0x148];                              /* 0x5000400520c39c01 */
        /*04d8*/                   DMUL R18, R14, R16;                                        /* 0x5000000040e49c01 */
        /*04e0*/                   DFMA R12, -R4, R18, R14;                                   /* 0x201c000048431e01 */
        /*04e8*/                   DFMA R16, R16, R12, R18;                                   /* 0x2024000031041c01 */
        /*04f0*/                   DFMA R6, R24, c[0x0][0x148], R6;                           /* 0x200c400521819c01 */
        /*04f8*/                   DFMA R8, R10, c[0x0][0x148], R8;                           /* 0x2010400520a21c01 */
                                                                                              /* 0x2202b2c042d20047 */
        /*0508*/                   FFMA R10, RZ, R5, R17;                                     /* 0x3022000017f29c00 */
        /*0510*/                   ST.E.64 [R2+0xfa18], R6;                                   /* 0x940003e860219ca5 */
        /*0518*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
        /*0520*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*0528*/                   ST.E.64 [R2+0x1f418], R8;                                  /* 0x940007d060221ca5 */
        /*0530*/               @P0 BRA CC.NEU, 0x568;                                         /* 0x40000000c00001a7 */
        /*0538*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
                                                                                              /* 0x223042b2b042b2b7 */
        /*0548*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
        /*0550*/                   CAL 0xa78;                                                 /* 0x5000001480011c07 */
        /*0558*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
        /*0560*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*0568*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0570*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
        /*0578*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
                                                                                              /* 0x22329283f2323047 */
        /*0588*/                   LD.E.64 R8, [R2+0x5dc18];                                  /* 0x8400177060221ca5 */
        /*0590*/                   SSY 0x638;                                                 /* 0x6000000280001c07 */
        /*0598*/                   LD.E.64 R6, [R2+0x2ee18];                                  /* 0x84000bb860219ca5 */
        /*05a0*/                   LD.E.64 R22, [R2+0x3e818];                                 /* 0x84000fa060259ca5 */
        /*05a8*/                   MUFU.RCP64H R11, R5;                                       /* 0xc80000001852dc00 */
        /*05b0*/                   DFMA R12, -R4, R10, c[0x2][0x0];                           /* 0x2014880000431e01 */
        /*05b8*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
                                                                                              /* 0x2232523292929257 */
        /*05c8*/                   DMUL R8, R8, c[0x0][0x148];                                /* 0x5000400520821c01 */
        /*05d0*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*05d8*/                   DMUL R14, R8, R10;                                         /* 0x5000000028839c01 */
        /*05e0*/                   DFMA R12, -R4, R14, R8;                                    /* 0x2010000038431e01 */
        /*05e8*/                   DFMA R10, R10, R12, R14;                                   /* 0x201c000030a29c01 */
        /*05f0*/                   DADD R24, R16, R6;                                         /* 0x4800000019061c01 */
        /*05f8*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
                                                                                              /* 0x2232b2b2b2c24047 */
        /*0608*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*0610*/                   ST.E.64 [R2+0x2ee18], R24;                                 /* 0x94000bb860261ca5 */
        /*0618*/                   FSETP.GT.AND P0, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003061dc80 */
        /*0620*/               @P0 BRA CC.NEU, 0x630;                                         /* 0x40000000200001a7 */
        /*0628*/                   CAL 0xa78;                                                 /* 0x5000001120011c07 */
        /*0630*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0638*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
                                                                                              /* 0x22e2e04230423047 */
        /*0648*/                   LD.E.64 R12, [R2+0x4e218];                                 /* 0x8400138860231ca5 */
        /*0650*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*0658*/                   LD.E.64 R6, [R2+0xfa18];                                   /* 0x840003e860219ca5 */
        /*0660*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0668*/                   LD.E.64 R8, [R2+0x1f418];                                  /* 0x840007d060221ca5 */
        /*0670*/                   SSY 0x758;                                                 /* 0x6000000380001c07 */
        /*0678*/                   ST.E.64 [R2+0x3e818], R22;                                 /* 0x94000fa060259ca5 */
                                                                                              /* 0x22929232929283f7 */
        /*0688*/                   LD.E.64 R10, [R2+0x3e818];                                 /* 0x84000fa060229ca5 */
        /*0690*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*0698*/                   DFMA R16, -R4, R14, c[0x2][0x0];                           /* 0x201c880000441e01 */
        /*06a0*/                   DFMA R16, R16, R16, R16;                                   /* 0x2020000041041c01 */
        /*06a8*/                   DFMA R16, R14, R16, R14;                                   /* 0x201c000040e41c01 */
        /*06b0*/                   DMUL R14, R12, c[0x0][0x148];                              /* 0x5000400520c39c01 */
        /*06b8*/                   DMUL R18, R14, R16;                                        /* 0x5000000040e49c01 */
                                                                                              /* 0x22d2004252323297 */
        /*06c8*/                   DFMA R12, -R4, R18, R14;                                   /* 0x201c000048431e01 */
        /*06d0*/                   DFMA R16, R16, R12, R18;                                   /* 0x2024000031041c01 */
        /*06d8*/                   DFMA R6, R24, c[0x0][0x148], R6;                           /* 0x200c400521819c01 */
        /*06e0*/                   DFMA R8, R10, c[0x0][0x148], R8;                           /* 0x2010400520a21c01 */
        /*06e8*/                   FFMA R10, RZ, R5, R17;                                     /* 0x3022000017f29c00 */
        /*06f0*/                   ST.E.64 [R2+0xfa18], R6;                                   /* 0x940003e860219ca5 */
        /*06f8*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
                                                                                              /* 0x2202b2b042b2c047 */
        /*0708*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*0710*/                   ST.E.64 [R2+0x1f418], R8;                                  /* 0x940007d060221ca5 */
        /*0718*/               @P0 BRA CC.NEU, 0x750;                                         /* 0x40000000c00001a7 */
        /*0720*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0728*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
        /*0730*/                   CAL 0xa78;                                                 /* 0x5000000d00011c07 */
        /*0738*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
                                                                                              /* 0x223230423042b2b7 */
        /*0748*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*0750*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0758*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
        /*0760*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*0768*/                   LD.E.64 R8, [R2+0x5dc18];                                  /* 0x8400177060221ca5 */
        /*0770*/                   SSY 0x820;                                                 /* 0x60000002a0001c07 */
        /*0778*/                   LD.E.64 R6, [R2+0x2ee18];                                  /* 0x84000bb860219ca5 */
                                                                                              /* 0x22929252329283f7 */
        /*0788*/                   LD.E.64 R22, [R2+0x3e818];                                 /* 0x84000fa060259ca5 */
        /*0790*/                   MUFU.RCP64H R11, R5;                                       /* 0xc80000001852dc00 */
        /*0798*/                   DFMA R12, -R4, R10, c[0x2][0x0];                           /* 0x2014880000431e01 */
        /*07a0*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*07a8*/                   DMUL R8, R8, c[0x0][0x148];                                /* 0x5000400520821c01 */
        /*07b0*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*07b8*/                   DMUL R14, R8, R10;                                         /* 0x5000000028839c01 */
                                                                                              /* 0x22c2720042923297 */
        /*07c8*/                   DFMA R12, -R4, R14, R8;                                    /* 0x2010000038431e01 */
        /*07d0*/                   DFMA R10, R10, R12, R14;                                   /* 0x201c000030a29c01 */
        /*07d8*/                   DADD R24, R16, R6;                                         /* 0x4800000019061c01 */
        /*07e0*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*07e8*/                   ST.E.64 [R2+0x2ee18], R24;                                 /* 0x94000bb860261ca5 */
        /*07f0*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*07f8*/                   FSETP.GT.AND P0, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003061dc80 */
                                                                                              /* 0x2230423042b2b2b7 */
        /*0808*/               @P0 BRA CC.NEU, 0x818;                                         /* 0x40000000200001a7 */
        /*0810*/                   CAL 0xa78;                                                 /* 0x5000000980011c07 */
        /*0818*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0820*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
        /*0828*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*0830*/                   LD.E.64 R12, [R2+0x4e218];                                 /* 0x8400138860231ca5 */
        /*0838*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
                                                                                              /* 0x229283f2e2e23047 */
        /*0848*/                   LD.E.64 R6, [R2+0xfa18];                                   /* 0x840003e860219ca5 */
        /*0850*/                   SSY 0x938;                                                 /* 0x6000000380001c07 */
        /*0858*/                   LD.E.64 R8, [R2+0x1f418];                                  /* 0x840007d060221ca5 */
        /*0860*/                   ST.E.64 [R2+0x3e818], R22;                                 /* 0x94000fa060259ca5 */
        /*0868*/                   LD.E.64 R10, [R2+0x3e818];                                 /* 0x84000fa060229ca5 */
        /*0870*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*0878*/                   DFMA R16, -R4, R14, c[0x2][0x0];                           /* 0x201c880000441e01 */
                                                                                              /* 0x2232329292923297 */
        /*0888*/                   DFMA R16, R16, R16, R16;                                   /* 0x2020000041041c01 */
        /*0890*/                   DFMA R16, R14, R16, R14;                                   /* 0x201c000040e41c01 */
        /*0898*/                   DMUL R14, R12, c[0x0][0x148];                              /* 0x5000400520c39c01 */
        /*08a0*/                   DMUL R18, R14, R16;                                        /* 0x5000000040e49c01 */
        /*08a8*/                   DFMA R12, -R4, R18, R14;                                   /* 0x201c000048431e01 */
        /*08b0*/                   DFMA R16, R16, R12, R18;                                   /* 0x2024000031041c01 */
        /*08b8*/                   DFMA R6, R24, c[0x0][0x148], R6;                           /* 0x200c400521819c01 */
                                                                                              /* 0x22b2c042d2004257 */
        /*08c8*/                   DFMA R8, R10, c[0x0][0x148], R8;                           /* 0x2010400520a21c01 */
        /*08d0*/                   FFMA R10, RZ, R5, R17;                                     /* 0x3022000017f29c00 */
        /*08d8*/                   ST.E.64 [R2+0xfa18], R6;                                   /* 0x940003e860219ca5 */
        /*08e0*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
        /*08e8*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*08f0*/                   ST.E.64 [R2+0x1f418], R8;                                  /* 0x940007d060221ca5 */
        /*08f8*/               @P0 BRA CC.NEU, 0x930;                                         /* 0x40000000c00001a7 */
                                                                                              /* 0x2232b2b042b2b047 */
        /*0908*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0910*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
        /*0918*/                   CAL 0xa78;                                                 /* 0x5000000560011c07 */
        /*0920*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
        /*0928*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*0930*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0938*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
                                                                                              /* 0x229283f230423047 */
        /*0948*/                   LD.E.64 R8, [R2+0x5dc18];                                  /* 0x8400177060221ca5 */
        /*0950*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*0958*/                   LD.E.64 R6, [R2+0x2ee18];                                  /* 0x84000bb860219ca5 */
        /*0960*/                   SSY 0xa00;                                                 /* 0x6000000260001c07 */
        /*0968*/                   LD.E.64 R22, [R2+0x3e818];                                 /* 0x84000fa060259ca5 */
        /*0970*/                   MUFU.RCP64H R11, R5;                                       /* 0xc80000001852dc00 */
        /*0978*/                   DFMA R12, -R4, R10, c[0x2][0x0];                           /* 0x2014880000431e01 */
                                                                                              /* 0x2292329292925237 */
        /*0988*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*0990*/                   DMUL R8, R8, c[0x0][0x148];                                /* 0x5000400520821c01 */
        /*0998*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*09a0*/                   DMUL R14, R8, R10;                                         /* 0x5000000028839c01 */
        /*09a8*/                   DFMA R12, -R4, R14, R8;                                    /* 0x2010000038431e01 */
        /*09b0*/                   DFMA R10, R10, R12, R14;                                   /* 0x201c000030a29c01 */
        /*09b8*/                   DADD R24, R16, R6;                                         /* 0x4800000019061c01 */
                                                                                              /* 0x22b2b2b2c2720047 */
        /*09c8*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*09d0*/                   ST.E.64 [R2+0x2ee18], R24;                                 /* 0x94000bb860261ca5 */
        /*09d8*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*09e0*/                   FSETP.GT.AND P0, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003061dc80 */
        /*09e8*/               @P0 BRA CC.NEU, 0x9f8;                                         /* 0x40000000200001a7 */
        /*09f0*/                   CAL 0xa78;                                                 /* 0x5000000200011c07 */
        /*09f8*/                   NOP.S;                                                     /* 0x4000000000001df4 */
                                                                                              /* 0x23f2e29042804207 */
        /*0a08*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*0a10*/                   LD.E.64 R4, [R2+0xfa18];                                   /* 0x840003e860211ca5 */
        /*0a18*/                   IADD32I R0, R0, 0x5;                                       /* 0x0800000014001c02 */
        /*0a20*/                   LD.E.64 R6, [R2+0x1f418];                                  /* 0x840007d060219ca5 */
        /*0a28*/                   ISETP.NE.AND P0, PT, R0, RZ, PT;                           /* 0x1a8e0000fc01dc23 */
        /*0a30*/                   ST.E.64 [R2+0x3e818], R22;                                 /* 0x94000fa060259ca5 */
        /*0a38*/                   LD.E.64 R8, [R2+0x3e818];                                  /* 0x84000fa060221ca5 */
                                                                                              /* 0x2202e2e042937297 */
        /*0a48*/                   DFMA R4, R24, c[0x0][0x148], R4;                           /* 0x2008400521811c01 */
        /*0a50*/                   ST.E.64 [R2+0xfa18], R4;                                   /* 0x940003e860211ca5 */
        /*0a58*/                   DFMA R6, R8, c[0x0][0x148], R6;                            /* 0x200c400520819c01 */
        /*0a60*/                   ST.E.64 [R2+0x1f418], R6;                                  /* 0x940007d060219ca5 */
        /*0a68*/               @P0 BRA 0xe0;                                                  /* 0x4003ffd9c00001e7 */
        /*0a70*/                   EXIT;                                                      /* 0x8000000000001de7 */
        /*0a78*/                   LOP32I.AND R6, R5, 0x40000000;                             /* 0x3900000000519c02 */
                                                                                              /* 0x22028042c0427047 */
        /*0a88*/                   MOV32I R12, 0x5ff00000;                                    /* 0x197fc00000031de2 */
        /*0a90*/                   LOP32I.AND R13, R9, 0x7f800000;                            /* 0x39fe000000935c02 */
        /*0a98*/                   ISETP.LT.U32.AND P0, PT, R6, c[0x2][0x10], PT;             /* 0x188e48004061dc03 */
        /*0aa0*/                   MOV R6, RZ;                                                /* 0x28000000fc019de4 */
        /*0aa8*/                   SEL R7, R12, c[0x2][0x14], P0;                             /* 0x2000480050c1dc04 */
        /*0ab0*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0ab8*/                   DMUL R10, R4, R6;                                          /* 0x5000000018429c01 */
                                                                                              /* 0x2232820280428047 */
        /*0ac8*/                   ISETP.LT.U32.AND P0, PT, R13, c[0x2][0x18], PT;            /* 0x188e480060d1dc03 */
        /*0ad0*/                   MOV R18, RZ;                                               /* 0x28000000fc049de4 */
        /*0ad8*/                   MUFU.RCP64H R15, R11;                                      /* 0xc800000018b3dc00 */
        /*0ae0*/                   PBK 0xd60;                                                 /* 0x68000009e0001c07 */
        /*0ae8*/                   DFMA R20, -R10, R14, c[0x2][0x0];                          /* 0x201c880000a51e01 */
        /*0af0*/                   SEL R19, R12, c[0x2][0x14], P0;                            /* 0x2000480050c4dc04 */
        /*0af8*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
                                                                                              /* 0x22b2d29292929257 */
        /*0b08*/                   DMUL R16, R8, R18;                                         /* 0x5000000048841c01 */
        /*0b10*/                   DFMA R14, R14, R20, R14;                                   /* 0x201c000050e39c01 */
        /*0b18*/                   DMUL R20, R16, R14;                                        /* 0x5000000039051c01 */
        /*0b20*/                   DFMA R10, -R10, R20, R16;                                  /* 0x2020000050a29e01 */
        /*0b28*/                   DFMA R10, R14, R10, R20;                                   /* 0x2028000028e29c01 */
        /*0b30*/                   DSETP.LEU.AND P0, PT, |R10|, RZ, PT;                       /* 0x1d8e0000fca1dc81 */
        /*0b38*/               @P0 BRA 0xce0;                                                 /* 0x40000006800001e7 */
                                                                                              /* 0x223252328202b207 */
        /*0b48*/                   ISETP.GT.U32.AND P0, PT, R13, c[0x2][0x1c], PT;            /* 0x1a0e480070d1dc03 */
        /*0b50*/                   DMUL R14, R6, R10;                                         /* 0x5000000028639c01 */
        /*0b58*/                   SEL R13, R12, c[0x2][0x14], P0;                            /* 0x2000480050c35c04 */
        /*0b60*/                   MOV R12, RZ;                                               /* 0x28000000fc031de4 */
        /*0b68*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0b70*/                   DMUL R10, R12, R14;                                        /* 0x5000000038c29c01 */
        /*0b78*/                   DMUL R6, R6, R16;                                          /* 0x5000000040619c01 */
                                                                                              /* 0x22b2c27202d29257 */
        /*0b88*/                   DFMA R12, R4, R10, -R8;                                    /* 0x2010000028431d01 */
        /*0b90*/                   DFMA R14, R4, R6, -R8;                                     /* 0x2010000018439d01 */
        /*0b98*/                   DSETP.GT.AND P0, PT, |R12|, |R14|, PT;                     /* 0x1a0e000038c1dcc1 */
        /*0ba0*/                   SEL R7, R7, R11, P0;                                       /* 0x200000002c71dc04 */
        /*0ba8*/                   SEL R6, R6, R10, P0;                                       /* 0x2000000028619c04 */
        /*0bb0*/                   FSETP.GTU.AND P0, PT, |R7|, 1.469367938527859385e-39, PT;  /* 0x260ec0040071dc80 */
        /*0bb8*/               @P0 BRK;                                                       /* 0xa8000000000001e7 */
                                                                                              /* 0x2282804220420047 */
        /*0bc8*/                   FSETP.LT.AND P0, PT, |R9|, 1.5046327690525280102e-36, PT;  /* 0x208ec1000091dc80 */
        /*0bd0*/                   LOP32I.AND R10, R6, 0xfffffffe;                            /* 0x3bfffffff8629c02 */
        /*0bd8*/                   MOV32I R15, 0x58500000;                                    /* 0x196140000003dde2 */
        /*0be0*/                   MOV R11, R7;                                               /* 0x280000001c02dde4 */
        /*0be8*/                   MOV R14, RZ;                                               /* 0x28000000fc039de4 */
        /*0bf0*/                   LOP32I.OR R6, R6, 0x1;                                     /* 0x3800000004619c42 */
        /*0bf8*/                   SEL R15, R15, c[0x2][0x4], P0;                             /* 0x2000480010f3dc04 */
                                                                                              /* 0x220282d292329237 */
        /*0c08*/                   DMUL R4, R4, R14;                                          /* 0x5000000038411c01 */
        /*0c10*/                   DMUL R8, R8, R14;                                          /* 0x5000000038821c01 */
        /*0c18*/                   DFMA R12, R10, R4, -R8;                                    /* 0x2010000010a31d01 */
        /*0c20*/                   DFMA R14, R6, R4, -R8;                                     /* 0x2010000010639d01 */
        /*0c28*/                   DSETP.GT.AND P0, PT, |R12|, |R14|, PT;                     /* 0x1a0e000038c1dcc1 */
        /*0c30*/                   SEL R11, R6, R10, P0;                                      /* 0x200000002862dc04 */
        /*0c38*/                   IADD32I R13.CC, R11, 0x1;                                  /* 0x0c00000004b35c02 */
                                                                                              /* 0x22004202b20042b7 */
        /*0c48*/                   LOP32I.AND R10, R11, 0x1;                                  /* 0x3800000004b29c02 */
        /*0c50*/                   ISETP.EQ.U32.AND P0, PT, R10, 0x1, PT;                     /* 0x190ec00004a1dc03 */
        /*0c58*/                   IADD.X R12, RZ, R7;                                        /* 0x480000001ff31c43 */
        /*0c60*/                   IADD32I R6.CC, R11, -0x1;                                  /* 0x0ffffffffcb19c02 */
        /*0c68*/                   SEL R13, R11, R13, !P0;                                    /* 0x2010000034b35c04 */
        /*0c70*/                   IADD32I.X R10, R7, -0x1;                                   /* 0x0bfffffffc729c42 */
        /*0c78*/                   SEL R12, R7, R12, !P0;                                     /* 0x2010000030731c04 */
                                                                                              /* 0x22d2923282020277 */
        /*0c88*/                   SEL R6, R6, R11, !P0;                                      /* 0x201000002c619c04 */
        /*0c90*/                   SEL R7, R10, R7, !P0;                                      /* 0x201000001ca1dc04 */
        /*0c98*/                   MOV R10, R13;                                              /* 0x2800000034029de4 */
        /*0ca0*/                   MOV R11, R12;                                              /* 0x280000003002dde4 */
        /*0ca8*/                   DFMA R10, R4, R10, -R8;                                    /* 0x2010000028429d01 */
        /*0cb0*/                   DFMA R4, R4, R6, -R8;                                      /* 0x2010000018411d01 */
        /*0cb8*/                   DSETP.GT.AND P0, PT, |R10|, |R4|, PT;                      /* 0x1a0e000010a1dcc1 */
                                                                                              /* 0x228042b2d2e04207 */
        /*0cc8*/                   SEL R6, R6, R13, P0;                                       /* 0x2000000034619c04 */
        /*0cd0*/                   SEL R7, R7, R12, P0;                                       /* 0x200000003071dc04 */
        /*0cd8*/                   BRK;                                                       /* 0xa800000000001de7 */
        /*0ce0*/                   DSETP.EQ.AND P0, PT, R10, RZ, PT;                          /* 0x190e0000fca1dc01 */
        /*0ce8*/               @P0 BRA 0xd50;                                                 /* 0x40000001800001e7 */
        /*0cf0*/                   MUFU.RCP64H R7, R5;                                        /* 0xc80000001851dc00 */
        /*0cf8*/                   MOV R6, RZ;                                                /* 0x28000000fc019de4 */
                                                                                              /* 0x220282027202d2d7 */
        /*0d08*/                   DSETP.GT.AND P1, PT, |R6|, RZ, PT;                         /* 0x1a0e0000fc63dc81 */
        /*0d10*/              @!P1 DSETP.NEU.AND P0, PT, |R4|, +INF , PT;                     /* 0x1e8edffc0041e481 */
        /*0d18*/              @!P1 SEL R4, R4, R6, P0;                                        /* 0x2000000018412404 */
        /*0d20*/              @!P1 SEL R5, R5, R7, P0;                                        /* 0x200000001c516404 */
        /*0d28*/              @!P1 MOV R6, R4;                                                /* 0x280000001001a5e4 */
        /*0d30*/              @!P1 MOV R7, R5;                                                /* 0x280000001401e5e4 */
        /*0d38*/                   DMUL R6, R8, R6;                                           /* 0x5000000018819c01 */
                                                                                              /* 0x2002e20042b202e7 */
        /*0d48*/                   BRK;                                                       /* 0xa800000000001de7 */
        /*0d50*/                   DMUL R6, R8, R4;                                           /* 0x5000000010819c01 */
        /*0d58*/                   BRK;                                                       /* 0xa800000000001de7 */
        /*0d60*/                   MOV R10, R6;                                               /* 0x2800000018029de4 */
        /*0d68*/                   MOV R11, R7;                                               /* 0x280000001c02dde4 */
        /*0d70*/                   RET;                                                       /* 0x9000000000001de7 */
        /*0d78*/                   BRA 0xd78;                                                 /* 0x4003ffffe0001de7 */
		...........................................................................................................................................................


		Function : _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEdEUldE_JdEEEvT_DpT0_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                               /* 0x22804280423042b7 */
        /*0008*/                   MOV R1, c[0x0][0x44];                                       /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R7, 0x0;                                             /* 0x180000000001dde2 */
        /*0018*/                   S2R R4, SR_TID.X;                                           /* 0x2c00000084011c04 */
        /*0020*/                   MOV32I R6, 0x0;                                             /* 0x1800000000019de2 */
        /*0028*/                   S2R R0, SR_CTAID.X;                                         /* 0x2c00000094001c04 */
        /*0030*/                   MOV R2, R7;                                                 /* 0x280000001c009de4 */
        /*0038*/                   MOV R3, R6;                                                 /* 0x280000001800dde4 */
                                                                                               /* 0x2282e2c2c3f28047 */
        /*0048*/                   LD.E.64 R2, [R2];                                           /* 0x8400000000209ca5 */
        /*0050*/                   IMAD R4, R0, c[0x0][0x28], R4;                              /* 0x20084000a0011ca3 */
        /*0058*/                   SHR R5, R4, 0x1f;                                           /* 0x5800c0007c415c23 */
        /*0060*/                   ISUB RZ.CC, R4, R2;                                         /* 0x48010000084fdd03 */
        /*0068*/                   ISETP.GE.U32.X.AND P0, PT, R5, R3, PT;                      /* 0x1b0e00000c51dc43 */
        /*0070*/               @P0 EXIT;                                                       /* 0x80000000000001e7 */
        /*0078*/                   MOV32I R0, 0x3;                                             /* 0x180000000c001de2 */
                                                                                               /* 0x2202404230423047 */
        /*0088*/                   IADD32I RZ.CC, R0, -0x20;                                   /* 0x0fffffff800fdc02 */
        /*0090*/                   SHL R5, R5, 0x3;                                            /* 0x6000c0000c515c03 */
        /*0098*/                   SHL R0, R4, -0x1d;                                          /* 0x6000ffff8c401c03 */
        /*00a0*/                   MOV32I R8, 0x1;                                             /* 0x1800000004021de2 */
        /*00a8*/                   SHR.U32 R2, R4, 0x1d;                                       /* 0x5800c00074409c03 */
        /*00b0*/                   SSY 0x1a0;                                                  /* 0x60000003a0001c07 */
        /*00b8*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
                                                                                               /* 0x2283f232820282b7 */
        /*00c8*/                   IMAD.U32.U32 R4.CC, R4, 0x8, R7;                            /* 0x200fc00020411c03 */
        /*00d0*/               @P0 IADD R0, R5, R2;                                            /* 0x4800000008500003 */
        /*00d8*/                   IADD.X R7, R0, R6;                                          /* 0x480000001801dc43 */
        /*00e0*/                   MOV R6, R4;                                                 /* 0x2800000010019de4 */
        /*00e8*/                   LD.E.64 R4, [R6+0x18];                                      /* 0x8400000060611ca5 */
        /*00f0*/                   LD.E.64 R2, [R6+0x4e218];                                   /* 0x8400138860609ca5 */
        /*00f8*/                   MUFU.RCP64H R9, R5;                                         /* 0xc800000018525c00 */
                                                                                               /* 0x2202929292523297 */
        /*0108*/                   DFMA R10, -R4, R8, c[0x2][0x0];                             /* 0x2010880000429e01 */
        /*0110*/                   DFMA R10, R10, R10, R10;                                    /* 0x2014000028a29c01 */
        /*0118*/                   DMUL R14, R2, c[0x0][0x148];                                /* 0x5000400520239c01 */
        /*0120*/                   DFMA R8, R8, R10, R8;                                       /* 0x2010000028821c01 */
        /*0128*/                   DMUL R2, R14, R8;                                           /* 0x5000000020e09c01 */
        /*0130*/                   DFMA R12, -R4, R2, R14;                                     /* 0x201c000008431e01 */
        /*0138*/                   DFMA R8, R8, R12, R2;                                       /* 0x2004000030821c01 */
                                                                                               /* 0x22b2b042b2c28287 */
        /*0148*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                 /* 0x160f480020ffdc80 */
        /*0150*/                   FFMA R0, RZ, R5, R9;                                        /* 0x3012000017f01c00 */
        /*0158*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0xc], PT;                 /* 0x220e48003001dc80 */
        /*0160*/               @P0 BRA CC.NEU, 0x198;                                          /* 0x40000000c00001a7 */
        /*0168*/                   MOV R10, R4;                                                /* 0x2800000010029de4 */
        /*0170*/                   MOV R11, R5;                                                /* 0x280000001402dde4 */
        /*0178*/                   CAL 0x330;                                                  /* 0x50000006c0011c07 */
                                                                                               /* 0x2230423042b2b047 */
        /*0188*/                   MOV R8, R22;                                                /* 0x2800000058021de4 */
        /*0190*/                   MOV R9, R23;                                                /* 0x280000005c025de4 */
        /*0198*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*01a0*/                   LD.E.64 R10, [R6+0x18];                                     /* 0x8400000060629ca5 */
        /*01a8*/                   MOV32I R16, 0x1;                                            /* 0x1800000004041de2 */
        /*01b0*/                   LD.E.64 R14, [R6+0x5dc18];                                  /* 0x8400177060639ca5 */
        /*01b8*/                   SSY 0x290;                                                  /* 0x6000000340001c07 */
                                                                                               /* 0x22329283f2323237 */
        /*01c8*/                   LD.E.64 R18, [R6+0x2ee18];                                  /* 0x84000bb860649ca5 */
        /*01d0*/                   LD.E.64 R12, [R6+0xfa18];                                   /* 0x840003e860631ca5 */
        /*01d8*/                   LD.E.64 R4, [R6+0x3e818];                                   /* 0x84000fa060611ca5 */
        /*01e0*/                   LD.E.64 R2, [R6+0x1f418];                                   /* 0x840007d060609ca5 */
        /*01e8*/                   MUFU.RCP64H R17, R11;                                       /* 0xc800000018b45c00 */
        /*01f0*/                   DFMA R20, -R10, R16, c[0x2][0x0];                           /* 0x2020880000a51e01 */
        /*01f8*/                   DFMA R20, R20, R20, R20;                                    /* 0x2028000051451c01 */
                                                                                               /* 0x2252325232929257 */
        /*0208*/                   DMUL R14, R14, c[0x0][0x148];                               /* 0x5000400520e39c01 */
        /*0210*/                   DFMA R16, R16, R20, R16;                                    /* 0x2020000051041c01 */
        /*0218*/                   DMUL R20, R14, R16;                                         /* 0x5000000040e51c01 */
        /*0220*/                   DFMA R22, -R10, R20, R14;                                   /* 0x201c000050a59e01 */
        /*0228*/                   DADD R18, R8, R18;                                          /* 0x4800000048849c01 */
        /*0230*/                   DFMA R22, R16, R22, R20;                                    /* 0x2028000059059c01 */
        /*0238*/                   DFMA R8, R18, c[0x0][0x148], R12;                           /* 0x2018400521221c01 */
                                                                                               /* 0x22b2b2c042d20047 */
        /*0248*/                   FFMA R0, RZ, R11, R23;                                      /* 0x302e00002ff01c00 */
        /*0250*/                   ST.E.64 [R6+0x2ee18], R18;                                  /* 0x94000bb860649ca5 */
        /*0258*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                 /* 0x160f480020ffdc80 */
        /*0260*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0xc], PT;                 /* 0x220e48003001dc80 */
        /*0268*/                   ST.E.64 [R6+0xfa18], R8;                                    /* 0x940003e860621ca5 */
        /*0270*/               @P0 BRA CC.NEU, 0x280;                                          /* 0x40000000200001a7 */
        /*0278*/                   CAL 0x330;                                                  /* 0x50000002c0011c07 */
                                                                                               /* 0x22725280420232b7 */
        /*0288*/                   DADD.S R4, R4, R22;                                         /* 0x4800000058411c11 */
        /*0290*/                   DSETP.GT.AND P0, PT, R8, 1, PT;                             /* 0x1a0ecffc0081dc01 */
        /*0298*/                   DFMA R2, R4, c[0x0][0x148], R2;                             /* 0x2004400520409c01 */
        /*02a0*/                   ST.E.64 [R6+0x3e818], R4;                                   /* 0x94000fa060611ca5 */
        /*02a8*/                   SSY 0x2f0;                                                  /* 0x6000000100001c07 */
        /*02b0*/                   DSETP.LT.OR P0, PT, R8, -1, P0;                             /* 0x18a0effc0081dc01 */
        /*02b8*/                   ST.E.64 [R6+0x1f418], R2;                                   /* 0x940007d060609ca5 */
                                                                                               /* 0x22d2d2b283f232b7 */
        /*02c8*/              @!P0 NOP.S;                                                      /* 0x40000000000021f4 */
        /*02d0*/                   LD.E.64 R4, [R6+0x2ee18];                                   /* 0x84000bb860611ca5 */
        /*02d8*/                   LD.E.64 R2, [R6+0x1f418];                                   /* 0x840007d060609ca5 */
        /*02e0*/                   F2F.F64.F64 R4, -R4;                                        /* 0x1000000011b11d04 */
        /*02e8*/                   ST.E.64.S [R6+0x2ee18], R4;                                 /* 0x94000bb860611cb5 */
        /*02f0*/                   DSETP.GT.AND P0, PT, R2, 1, PT;                             /* 0x1a0ecffc0021dc01 */
        /*02f8*/                   DSETP.LT.OR P0, PT, R2, -1, P0;                             /* 0x18a0effc0021dc01 */
                                                                                               /* 0x228042e04283f2e7 */
        /*0308*/              @!P0 EXIT;                                                       /* 0x80000000000021e7 */
        /*0310*/                   LD.E.64 R2, [R6+0x3e818];                                   /* 0x84000fa060609ca5 */
        /*0318*/                   F2F.F64.F64 R2, -R2;                                        /* 0x1000000009b09d04 */
        /*0320*/                   ST.E.64 [R6+0x3e818], R2;                                   /* 0x94000fa060609ca5 */
        /*0328*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*0330*/                   LOP32I.AND R0, R11, 0x40000000;                             /* 0x3900000000b01c02 */
        /*0338*/                   LOP32I.AND R26, R15, 0x7f800000;                            /* 0x39fe000000f69c02 */
                                                                                               /* 0x22028042b0420047 */
        /*0348*/                   ISETP.LT.U32.AND P0, PT, R0, c[0x2][0x10], PT;              /* 0x188e48004001dc03 */
        /*0350*/                   MOV32I R0, 0x5ff00000;                                      /* 0x197fc00000001de2 */
        /*0358*/                   MOV R12, RZ;                                                /* 0x28000000fc031de4 */
        /*0360*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
        /*0368*/                   SEL R13, R0, c[0x2][0x14], P0;                              /* 0x2000480050035c04 */
        /*0370*/                   MOV R24, RZ;                                                /* 0x28000000fc061de4 */
        /*0378*/                   DMUL R16, R10, R12;                                         /* 0x5000000030a41c01 */
                                                                                               /* 0x225232928042c047 */
        /*0388*/                   ISETP.LT.U32.AND P0, PT, R26, c[0x2][0x18], PT;             /* 0x188e480061a1dc03 */
        /*0390*/                   PBK 0x628;                                                  /* 0x6800000a40001c07 */
        /*0398*/                   MUFU.RCP64H R19, R17;                                       /* 0xc80000001914dc00 */
        /*03a0*/                   SEL R25, R0, c[0x2][0x14], P0;                              /* 0x2000480050065c04 */
        /*03a8*/                   DFMA R20, -R16, R18, c[0x2][0x0];                           /* 0x2024880001051e01 */
        /*03b0*/                   DFMA R22, R20, R20, R20;                                    /* 0x2028000051459c01 */
        /*03b8*/                   DMUL R20, R14, R24;                                         /* 0x5000000060e51c01 */
                                                                                               /* 0x2202b2d292929297 */
        /*03c8*/                   DFMA R18, R18, R22, R18;                                    /* 0x2024000059249c01 */
        /*03d0*/                   DMUL R22, R20, R18;                                         /* 0x5000000049459c01 */
        /*03d8*/                   DFMA R16, -R16, R22, R20;                                   /* 0x2028000059041e01 */
        /*03e0*/                   DFMA R22, R18, R16, R22;                                    /* 0x202c000041259c01 */
        /*03e8*/                   DSETP.LEU.AND P0, PT, |R22|, RZ, PT;                        /* 0x1d8e0000fd61dc81 */
        /*03f0*/               @P0 BRA 0x5a8;                                                  /* 0x40000006c00001e7 */
        /*03f8*/                   ISETP.GT.U32.AND P0, PT, R26, c[0x2][0x1c], PT;             /* 0x1a0e480071a1dc03 */
                                                                                               /* 0x225232523282b047 */
        /*0408*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*0410*/                   DMUL R18, R12, R22;                                         /* 0x5000000058c49c01 */
        /*0418*/                   SEL R17, R0, c[0x2][0x14], P0;                              /* 0x2000480050045c04 */
        /*0420*/                   DMUL R22, R22, R16;                                         /* 0x5000000041659c01 */
        /*0428*/                   DMUL R16, R16, R18;                                         /* 0x5000000049041c01 */
        /*0430*/                   DMUL R12, R12, R22;                                         /* 0x5000000058c31c01 */
        /*0438*/                   DFMA R18, R10, R16, -R14;                                   /* 0x201c000040a49d01 */
                                                                                               /* 0x2202b2c27202d297 */
        /*0448*/                   DFMA R20, R10, R12, -R14;                                   /* 0x201c000030a51d01 */
        /*0450*/                   DSETP.GT.AND P0, PT, |R18|, |R20|, PT;                      /* 0x1a0e00005121dcc1 */
        /*0458*/                   SEL R13, R13, R17, P0;                                      /* 0x2000000044d35c04 */
        /*0460*/                   SEL R12, R12, R16, P0;                                      /* 0x2000000040c31c04 */
        /*0468*/                   FSETP.GTU.AND P0, PT, |R13|, 1.469367938527859385e-39, PT;  /* 0x260ec00400d1dc80 */
        /*0470*/               @P0 BRK;                                                        /* 0xa8000000000001e7 */
        /*0478*/                   FSETP.LT.AND P0, PT, |R15|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000f1dc80 */
                                                                                               /* 0x2232804270423047 */
        /*0488*/                   MOV32I R17, 0x58500000;                                     /* 0x1961400000045de2 */
        /*0490*/                   LOP32I.AND R0, R12, 0xfffffffe;                             /* 0x3bfffffff8c01c02 */
        /*0498*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*04a0*/                   LOP32I.OR R20, R12, 0x1;                                    /* 0x3800000004c51c42 */
        /*04a8*/                   SEL R17, R17, c[0x2][0x4], P0;                              /* 0x2000480011145c04 */
        /*04b0*/                   MOV R12, R0;                                                /* 0x2800000000031de4 */
        /*04b8*/                   DMUL R10, R10, R16;                                         /* 0x5000000040a29c01 */
                                                                                               /* 0x2282d29232720207 */
        /*04c8*/                   DMUL R14, R14, R16;                                         /* 0x5000000040e39c01 */
        /*04d0*/                   MOV R16, R20;                                               /* 0x2800000050041de4 */
        /*04d8*/                   MOV R17, R13;                                               /* 0x2800000034045de4 */
        /*04e0*/                   DFMA R18, R12, R10, -R14;                                   /* 0x201c000028c49d01 */
        /*04e8*/                   DFMA R16, R16, R10, -R14;                                   /* 0x201c000029041d01 */
        /*04f0*/                   DSETP.GT.AND P0, PT, |R18|, |R16|, PT;                      /* 0x1a0e00004121dcc1 */
        /*04f8*/                   SEL R20, R20, R0, P0;                                       /* 0x2000000001451c04 */
                                                                                               /* 0x220042c20042c047 */
        /*0508*/                   IADD32I R19.CC, R20, 0x1;                                   /* 0x0c0000000544dc02 */
        /*0510*/                   LOP32I.AND R12, R20, 0x1;                                   /* 0x3800000005431c02 */
        /*0518*/                   ISETP.EQ.U32.AND P0, PT, R12, 0x1, PT;                      /* 0x190ec00004c1dc03 */
        /*0520*/                   IADD.X R0, RZ, R13;                                         /* 0x4800000037f01c43 */
        /*0528*/                   IADD32I R16.CC, R20, -0x1;                                  /* 0x0ffffffffd441c02 */
        /*0530*/                   SEL R19, R20, R19, !P0;                                     /* 0x201000004d44dc04 */
        /*0538*/                   IADD32I.X R17, R13, -0x1;                                   /* 0x0bfffffffcd45c42 */
                                                                                               /* 0x2292328202026207 */
        /*0548*/                   SEL R0, R13, R0, !P0;                                       /* 0x2010000000d01c04 */
        /*0550*/                   SEL R16, R16, R20, !P0;                                     /* 0x2010000051041c04 */
        /*0558*/                   SEL R17, R17, R13, !P0;                                     /* 0x2010000035145c04 */
        /*0560*/                   MOV R12, R19;                                               /* 0x280000004c031de4 */
        /*0568*/                   MOV R13, R0;                                                /* 0x2800000000035de4 */
        /*0570*/                   DFMA R12, R10, R12, -R14;                                   /* 0x201c000030a31d01 */
        /*0578*/                   DFMA R10, R10, R16, -R14;                                   /* 0x201c000040a29d01 */
                                                                                               /* 0x2202b2d2e04202d7 */
        /*0588*/                   DSETP.GT.AND P0, PT, |R12|, |R10|, PT;                      /* 0x1a0e000028c1dcc1 */
        /*0590*/                   SEL R12, R16, R19, P0;                                      /* 0x200000004d031c04 */
        /*0598*/                   SEL R13, R17, R0, P0;                                       /* 0x2000000001135c04 */
        /*05a0*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*05a8*/                   DSETP.EQ.AND P0, PT, R22, RZ, PT;                           /* 0x190e0000fd61dc01 */
        /*05b0*/               @P0 BRA 0x618;                                                  /* 0x40000001800001e7 */
        /*05b8*/                   MUFU.RCP64H R13, R11;                                       /* 0xc800000018b35c00 */
                                                                                               /* 0x2282027202d2d287 */
        /*05c8*/                   MOV R12, RZ;                                                /* 0x28000000fc031de4 */
        /*05d0*/                   DSETP.GT.AND P1, PT, |R12|, RZ, PT;                         /* 0x1a0e0000fcc3dc81 */
        /*05d8*/              @!P1 DSETP.NEU.AND P0, PT, |R10|, +INF , PT;                     /* 0x1e8edffc00a1e481 */
        /*05e0*/              @!P1 SEL R0, R10, R12, P0;                                       /* 0x2000000030a02404 */
        /*05e8*/              @!P1 SEL R11, R11, R13, P0;                                      /* 0x2000000034b2e404 */
        /*05f0*/              @!P1 MOV R12, R0;                                                /* 0x28000000000325e4 */
        /*05f8*/              @!P1 MOV R13, R11;                                               /* 0x280000002c0365e4 */
                                                                                               /* 0x22e20042b202e207 */
        /*0608*/                   DMUL R12, R14, R12;                                         /* 0x5000000030e31c01 */
        /*0610*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*0618*/                   DMUL R12, R14, R10;                                         /* 0x5000000028e31c01 */
        /*0620*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*0628*/                   MOV R22, R12;                                               /* 0x2800000030059de4 */
        /*0630*/                   MOV R23, R13;                                               /* 0x280000003405dde4 */
        /*0638*/                   RET;                                                        /* 0x9000000000001de7 */
        /*0640*/                   BRA 0x640;                                                  /* 0x4003ffffe0001de7 */
        /*0648*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*0650*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*0658*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*0660*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*0668*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*0670*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*0678*/                   NOP;                                                        /* 0x4000000000001de4 */
		..........................................................................................................................................


		Function : _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_22add_all_forces_to_thisEvEEE4callEvEUlvE_JEEEvT_DpT0_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                               /* 0x22804280423042b7 */
        /*0008*/                   MOV R1, c[0x0][0x44];                                       /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R4, 0x0;                                             /* 0x1800000000011de2 */
        /*0018*/                   S2R R42, SR_TID.X;                                          /* 0x2c000000840a9c04 */
        /*0020*/                   MOV32I R5, 0x0;                                             /* 0x1800000000015de2 */
        /*0028*/                   S2R R3, SR_CTAID.X;                                         /* 0x2c0000009400dc04 */
        /*0030*/                   LD.E.64 R6, [R4];                                           /* 0x8400000000419ca5 */
        /*0038*/                   IMAD R42, R3, c[0x0][0x28], R42;                            /* 0x20544000a03a9ca3 */
                                                                                               /* 0x22028042e2c2c3f7 */
        /*0048*/                   SHR R2, R42, 0x1f;                                          /* 0x5800c0007ea09c23 */
        /*0050*/                   ISUB RZ.CC, R42, R6;                                        /* 0x480100001aafdd03 */
        /*0058*/                   ISETP.GE.U32.X.AND P0, PT, R2, R7, PT;                      /* 0x1b0e00001c21dc43 */
        /*0060*/               @P0 EXIT;                                                       /* 0x80000000000001e7 */
        /*0068*/                   MOV32I R0, 0x3;                                             /* 0x180000000c001de2 */
        /*0070*/                   SHL R3, R2, 0x3;                                            /* 0x6000c0000c20dc03 */
        /*0078*/                   IADD32I RZ.CC, R0, -0x20;                                   /* 0x0fffffff800fdc02 */
                                                                                               /* 0x22b2004270423047 */
        /*0088*/                   SHL R0, R42, -0x1d;                                         /* 0x6000ffff8ea01c03 */
        /*0090*/                   MOV32I R44, 0x1f40;                                         /* 0x1800007d000b1de2 */
        /*0098*/                   SHR.U32 R6, R42, 0x1d;                                      /* 0x5800c00076a19c03 */
        /*00a0*/                   MOV R43, RZ;                                                /* 0x28000000fc0adde4 */
        /*00a8*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*00b0*/                   MOV32I R8, 0x5;                                             /* 0x1800000014021de2 */
        /*00b8*/                   IMAD.U32.U32 R9.CC, R42, 0x8, R4;                           /* 0x2009c00022a25c03 */
                                                                                               /* 0x2202020200428047 */
        /*00c8*/               @P0 IADD R0, R3, R6;                                            /* 0x4800000018300003 */
        /*00d0*/                   MOV R37, RZ;                                                /* 0x28000000fc095de4 */
        /*00d8*/                   IADD.X R3, R0, R5;                                          /* 0x480000001400dc43 */
        /*00e0*/                   MOV32I R7, 0x4;                                             /* 0x180000001001dde2 */
        /*00e8*/                   IADD32I R33.CC, R42, 0x1;                                   /* 0x0c00000006a85c02 */
        /*00f0*/                   MOV R38, RZ;                                                /* 0x28000000fc099de4 */
        /*00f8*/                   MOV32I R6, 0x3;                                             /* 0x180000000c019de2 */
                                                                                               /* 0x22b0427200420047 */
        /*0108*/                   MOV R39, RZ;                                                /* 0x28000000fc09dde4 */
        /*0110*/                   MOV32I R0, 0x2;                                             /* 0x1800000008001de2 */
        /*0118*/                   MOV R40, RZ;                                                /* 0x28000000fc0a1de4 */
        /*0120*/                   MOV32I R32, 0x1;                                            /* 0x1800000004081de2 */
        /*0128*/                   MOV R41, RZ;                                                /* 0x28000000fc0a5de4 */
        /*0130*/                   IADD.X R36, RZ, R2;                                         /* 0x480000000bf91c43 */
        /*0138*/                   MOV R2, R9;                                                 /* 0x2800000024009de4 */
                                                                                               /* 0x22028042b2c2c047 */
        /*0148*/                   ISUB RZ.CC, R32, R33;                                       /* 0x48010000860fdd03 */
        /*0150*/                   SSY 0x5a8;                                                  /* 0x6000001140001c07 */
        /*0158*/                   ISETP.EQ.X.AND P0, PT, R41, R36, PT;                        /* 0x190e00009291dc63 */
        /*0160*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*0168*/                   MOV32I R9, 0x3;                                             /* 0x180000000c025de2 */
        /*0170*/                   SHL R10, R32, -0x1d;                                        /* 0x6000ffff8e029c03 */
        /*0178*/                   IADD32I RZ.CC, R9, -0x20;                                   /* 0x0fffffff809fdc02 */
                                                                                               /* 0x22b2004270423047 */
        /*0188*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
        /*0190*/                   SHL R11, R41, 0x3;                                          /* 0x6000c0000e92dc03 */
        /*0198*/                   SHR.U32 R12, R32, 0x1d;                                     /* 0x5800c00076031c03 */
        /*01a0*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*01a8*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*01b0*/                   LD.E.64 R24, [R2+0x1f418];                                  /* 0x840007d060261ca5 */
        /*01b8*/                   IMAD.U32.U32 R14.CC, R32, 0x8, R18;                         /* 0x2025c00022039c03 */
                                                                                               /* 0x2232804270420047 */
        /*01c8*/               @P0 IADD R10, R11, R12;                                         /* 0x4800000030b28003 */
        /*01d0*/                   LD.E.64 R26, [R2+0xfa18];                                   /* 0x840003e860269ca5 */
        /*01d8*/                   MOV R12, R14;                                               /* 0x2800000038031de4 */
        /*01e0*/                   MOV32I R20, 0x0;                                            /* 0x1800000000051de2 */
        /*01e8*/                   IADD.X R13, R10, R9;                                        /* 0x4800000024a35c43 */
        /*01f0*/                   MOV32I R21, 0x3fe00000;                                     /* 0x18ff800000055de2 */
        /*01f8*/                   LD.E.64 R10, [R12+0x1f410];                                 /* 0x840007d040c29ca5 */
                                                                                               /* 0x220292925233f047 */
        /*0208*/                   LD.E.64 R14, [R12+0xfa10];                                  /* 0x840003e840c39ca5 */
        /*0210*/                   SSY 0x2f8;                                                  /* 0x6000000380001c07 */
        /*0218*/                   DADD R24, -R24, R10;                                        /* 0x4800000029861e01 */
        /*0220*/                   DADD R26, -R26, R14;                                        /* 0x4800000039a69e01 */
        /*0228*/                   DMUL R14, R24, R24;                                         /* 0x5000000061839c01 */
        /*0230*/                   DFMA R10, R26, R26, R14;                                    /* 0x201c000069a29c01 */
        /*0238*/                   IADD32I R19, R11, -0x3500000;                               /* 0x0bf2c00000b4dc02 */
                                                                                               /* 0x2202923292928277 */
        /*0248*/                   MUFU.RSQ64H R15, R11;                                       /* 0xc80000001cb3dc00 */
        /*0250*/                   MOV R14, R19;                                               /* 0x280000004c039de4 */
        /*0258*/                   DMUL R16, R14, R14;                                         /* 0x5000000038e41c01 */
        /*0260*/                   DFMA R16, -R10, R16, c[0x2][0x0];                           /* 0x2020880000a41e01 */
        /*0268*/                   DFMA R12, R16, 0.375, R20;                                  /* 0x2028cff601031c01 */
        /*0270*/                   DMUL R16, R14, R16;                                         /* 0x5000000040e41c01 */
        /*0278*/                   DFMA R20, R12, R16, R14;                                    /* 0x201c000040c51c01 */
                                                                                               /* 0x2202820280420287 */
        /*0288*/                   ISETP.LT.U32.AND P0, PT, R19, c[0x2][0x8], PT;              /* 0x188e48002131dc03 */
        /*0290*/                   DMUL R16, R10, R20;                                         /* 0x5000000050a41c01 */
        /*0298*/                   IADD32I R23, R21, -0x100000;                                /* 0x0bffc0000155dc02 */
        /*02a0*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*02a8*/                   DFMA R12, -R16, R16, R10;                                   /* 0x2014000041031e01 */
        /*02b0*/                   MOV R15, R23;                                               /* 0x280000005c03dde4 */
        /*02b8*/                   DFMA R28, R12, R14, R16;                                    /* 0x2020000038c71c01 */
                                                                                               /* 0x22c2b2b2b20042b7 */
        /*02c8*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*02d0*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*02d8*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*02e0*/                   MOV R19, R23;                                               /* 0x280000005c04dde4 */
        /*02e8*/                   CAL 0x1b60;                                                 /* 0x50000061c0011c07 */
        /*02f0*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*02f8*/                   IMAD R20.CC, R42, 0x8, R18;                                 /* 0x2025c00022a51ca3 */
                                                                                               /* 0x2282804200428047 */
        /*0308*/                   IMAD.HI.X R21, R42, 0x8, R9;                                /* 0x2092c00022a55ce3 */
        /*0310*/                   LD.E.64 R14, [R4+0x18];                                     /* 0x8400000060439ca5 */
        /*0318*/                   LD.E.64 R10, [R20+0x18];                                    /* 0x8400000061429ca5 */
        /*0320*/                   DFMA R12, R28, R28, c[0x2][0x18];                           /* 0x2038880061c31c01 */
        /*0328*/                   MOV32I R16, 0x1;                                            /* 0x1800000004041de2 */
        /*0330*/                   SSY 0x3f0;                                                  /* 0x60000002e0001c07 */
        /*0338*/                   MUFU.RCP64H R17, R13;                                       /* 0xc800000018d45c00 */
                                                                                               /* 0x2292929293f29297 */
        /*0348*/                   DFMA R18, -R12, R16, c[0x2][0x0];                           /* 0x2020880000c49e01 */
        /*0350*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*0358*/                   DFMA R16, R16, R18, R16;                                    /* 0x2020000049041c01 */
        /*0360*/                   DMUL R10, R10, c[0x2][0x10];                                /* 0x5000480040a29c01 */
        /*0368*/                   DMUL R18, R10, R14;                                         /* 0x5000000038a49c01 */
        /*0370*/                   DMUL R30, R18, R16;                                         /* 0x5000000041279c01 */
        /*0378*/                   DFMA R10, -R12, R30, R18;                                   /* 0x2024000078c29e01 */
                                                                                               /* 0x220042b2c2828207 */
        /*0388*/                   DFMA R30, R16, R10, R30;                                    /* 0x203c000029079c01 */
        /*0390*/                   FSET.GTU.AND RZ.CC, |R19|, c[0x2][0xc], PT;                 /* 0x160f4800313fdc80 */
        /*0398*/                   FFMA R9, RZ, R13, R31;                                      /* 0x303e000037f25c00 */
        /*03a0*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*03a8*/               @P0 BRA CC.NEU, 0x3e8;                                          /* 0x40000000e00001a7 */
        /*03b0*/                   MOV R10, R18;                                               /* 0x2800000048029de4 */
        /*03b8*/                   MOV R11, R19;                                               /* 0x280000004c02dde4 */
                                                                                               /* 0x228042b2b042b2b7 */
        /*03c8*/                   MOV R9, R13;                                                /* 0x2800000034025de4 */
        /*03d0*/                   CAL 0x17e0;                                                 /* 0x5000005020011c07 */
        /*03d8*/                   MOV R30, R9;                                                /* 0x2800000024079de4 */
        /*03e0*/                   MOV R31, R10;                                               /* 0x280000002807dde4 */
        /*03e8*/                   MUFU.RCP64H.S R11, R29;                                     /* 0xc800000019d2dc10 */
        /*03f0*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*03f8*/                   DMUL R16, R26, R30;                                         /* 0x5000000079a41c01 */
                                                                                               /* 0x2282029292929297 */
        /*0408*/                   DFMA R12, -R28, R10, c[0x2][0x0];                           /* 0x2014880001c31e01 */
        /*0410*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*0418*/                   DFMA R10, R10, R12, R10;                                    /* 0x2014000030a29c01 */
        /*0420*/                   DMUL R14, R16, R10;                                         /* 0x5000000029039c01 */
        /*0428*/                   DFMA R12, -R28, R14, R16;                                   /* 0x2020000039c31e01 */
        /*0430*/                   DFMA R10, R10, R12, R14;                                    /* 0x201c000030a29c01 */
        /*0438*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0xc], PT;                 /* 0x160f4800311fdc80 */
                                                                                               /* 0x22020042b2c28047 */
        /*0448*/                   FFMA R9, RZ, R29, R11;                                      /* 0x3016000077f25c00 */
        /*0450*/                   SSY 0x4b0;                                                  /* 0x6000000160001c07 */
        /*0458*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*0460*/               @P0 BRA CC.NEU, 0x4a8;                                          /* 0x40000001000001a7 */
        /*0468*/                   MOV R10, R16;                                               /* 0x2800000040029de4 */
        /*0470*/                   MOV R11, R17;                                               /* 0x280000004402dde4 */
        /*0478*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
                                                                                               /* 0x223042b2b042b2b7 */
        /*0488*/                   MOV R9, R29;                                                /* 0x2800000074025de4 */
        /*0490*/                   CAL 0x17e0;                                                 /* 0x5000004d20011c07 */
        /*0498*/                   MOV R11, R10;                                               /* 0x280000002802dde4 */
        /*04a0*/                   MOV R10, R9;                                                /* 0x2800000024029de4 */
        /*04a8*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*04b0*/                   LD.E.64 R12, [R2+0x4e218];                                  /* 0x8400138860231ca5 */
        /*04b8*/                   MUFU.RCP64H R15, R29;                                       /* 0xc800000019d3dc00 */
                                                                                               /* 0x2292929280420047 */
        /*04c8*/                   LD.E.64 R26, [R2+0x5dc18];                                  /* 0x8400177060269ca5 */
        /*04d0*/                   DMUL R24, R24, R30;                                         /* 0x5000000079861c01 */
        /*04d8*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*04e0*/                   SSY 0x5a0;                                                  /* 0x60000002e0001c07 */
        /*04e8*/                   DFMA R16, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c41e01 */
        /*04f0*/                   DFMA R16, R16, R16, R16;                                    /* 0x2020000041041c01 */
        /*04f8*/                   DFMA R18, R14, R16, R14;                                    /* 0x201c000040e49c01 */
                                                                                               /* 0x2293f28282029297 */
        /*0508*/                   DMUL R14, R18, R24;                                         /* 0x5000000061239c01 */
        /*0510*/                   DFMA R16, -R28, R14, R24;                                   /* 0x2030000039c41e01 */
        /*0518*/                   DFMA R16, R18, R16, R14;                                    /* 0x201c000041241c01 */
        /*0520*/                   FSET.GTU.AND RZ.CC, |R25|, c[0x2][0xc], PT;                 /* 0x160f4800319fdc80 */
        /*0528*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*0530*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*0538*/                   DADD R12, R10, R12;                                         /* 0x4800000030a31c01 */
                                                                                               /* 0x22b2b0420042b047 */
        /*0548*/                   ST.E.64 [R2+0x4e218], R12;                                  /* 0x9400138860231ca5 */
        /*0550*/               @P0 BRA CC.NEU, 0x598;                                          /* 0x40000001000001a7 */
        /*0558*/                   MOV R10, R24;                                               /* 0x2800000060029de4 */
        /*0560*/                   MOV R11, R25;                                               /* 0x280000006402dde4 */
        /*0568*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*0570*/                   MOV R9, R29;                                                /* 0x2800000074025de4 */
        /*0578*/                   CAL 0x17e0;                                                 /* 0x5000004980011c07 */
                                                                                               /* 0x22c2c042b2b2b047 */
        /*0588*/                   MOV R16, R9;                                                /* 0x2800000024041de4 */
        /*0590*/                   MOV R17, R10;                                               /* 0x2800000028045de4 */
        /*0598*/                   DADD.S R16, R26, R16;                                       /* 0x4800000041a41c11 */
        /*05a0*/                   ST.E.64.S [R2+0x5dc18], R16;                                /* 0x9400177060241cb5 */
        /*05a8*/                   ISUB RZ.CC, R0, R33;                                        /* 0x48010000840fdd03 */
        /*05b0*/                   SSY 0xa10;                                                  /* 0x6000001160001c07 */
        /*05b8*/                   ISETP.EQ.X.AND P0, PT, R40, R36, PT;                        /* 0x190e00009281dc63 */
                                                                                               /* 0x22204200428042b7 */
        /*05c8*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*05d0*/                   MOV32I R9, 0x3;                                             /* 0x180000000c025de2 */
        /*05d8*/                   SHL R10, R0, -0x1d;                                         /* 0x6000ffff8c029c03 */
        /*05e0*/                   IADD32I RZ.CC, R9, -0x20;                                   /* 0x0fffffff809fdc02 */
        /*05e8*/                   SHL R11, R40, 0x3;                                          /* 0x6000c0000e82dc03 */
        /*05f0*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
        /*05f8*/                   LD.E.64 R24, [R2+0x1f418];                                  /* 0x840007d060261ca5 */
                                                                                               /* 0x2202b04200428047 */
        /*0608*/                   SHR.U32 R12, R0, 0x1d;                                      /* 0x5800c00074031c03 */
        /*0610*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*0618*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*0620*/                   LD.E.64 R26, [R2+0xfa18];                                   /* 0x840003e860269ca5 */
        /*0628*/                   IMAD.U32.U32 R14.CC, R0, 0x8, R18;                          /* 0x2025c00020039c03 */
        /*0630*/                   MOV32I R20, 0x0;                                            /* 0x1800000000051de2 */
        /*0638*/               @P0 IADD R10, R11, R12;                                         /* 0x4800000030b28003 */
                                                                                               /* 0x2233f23280427047 */
        /*0648*/                   MOV R12, R14;                                               /* 0x2800000038031de4 */
        /*0650*/                   MOV32I R21, 0x3fe00000;                                     /* 0x18ff800000055de2 */
        /*0658*/                   IADD.X R13, R10, R9;                                        /* 0x4800000024a35c43 */
        /*0660*/                   SSY 0x760;                                                  /* 0x60000003e0001c07 */
        /*0668*/                   LD.E.64 R10, [R12+0x1f410];                                 /* 0x840007d040c29ca5 */
        /*0670*/                   LD.E.64 R14, [R12+0xfa10];                                  /* 0x840003e840c39ca5 */
        /*0678*/                   DADD R24, -R24, R10;                                        /* 0x4800000029861e01 */
                                                                                               /* 0x2292828042929257 */
        /*0688*/                   DADD R26, -R26, R14;                                        /* 0x4800000039a69e01 */
        /*0690*/                   DMUL R14, R24, R24;                                         /* 0x5000000061839c01 */
        /*0698*/                   DFMA R10, R26, R26, R14;                                    /* 0x201c000069a29c01 */
        /*06a0*/                   IADD32I R19, R11, -0x3500000;                               /* 0x0bf2c00000b4dc02 */
        /*06a8*/                   MUFU.RSQ64H R15, R11;                                       /* 0xc80000001cb3dc00 */
        /*06b0*/                   MOV R14, R19;                                               /* 0x280000004c039de4 */
        /*06b8*/                   DMUL R16, R14, R14;                                         /* 0x5000000038e41c01 */
                                                                                               /* 0x2202028202923297 */
        /*06c8*/                   DFMA R16, -R10, R16, c[0x2][0x0];                           /* 0x2020880000a41e01 */
        /*06d0*/                   DFMA R12, R16, 0.375, R20;                                  /* 0x2028cff601031c01 */
        /*06d8*/                   DMUL R16, R14, R16;                                         /* 0x5000000040e41c01 */
        /*06e0*/                   DFMA R20, R12, R16, R14;                                    /* 0x201c000040c51c01 */
        /*06e8*/                   ISETP.LT.U32.AND P0, PT, R19, c[0x2][0x8], PT;              /* 0x188e48002131dc03 */
        /*06f0*/                   DMUL R16, R10, R20;                                         /* 0x5000000050a41c01 */
        /*06f8*/                   IADD32I R23, R21, -0x100000;                                /* 0x0bffc0000155dc02 */
                                                                                               /* 0x220042b202820277 */
        /*0708*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*0710*/                   DFMA R12, -R16, R16, R10;                                   /* 0x2014000041031e01 */
        /*0718*/                   MOV R15, R23;                                               /* 0x280000005c03dde4 */
        /*0720*/                   DFMA R28, R12, R14, R16;                                    /* 0x2020000038c71c01 */
        /*0728*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*0730*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*0738*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
                                                                                               /* 0x228042c042b2b2b7 */
        /*0748*/                   MOV R19, R23;                                               /* 0x280000005c04dde4 */
        /*0750*/                   CAL 0x1b60;                                                 /* 0x5000005020011c07 */
        /*0758*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0760*/                   IMAD R20.CC, R42, 0x8, R18;                                 /* 0x2025c00022a51ca3 */
        /*0768*/                   LD.E.64 R14, [R4+0x20];                                     /* 0x8400000080439ca5 */
        /*0770*/                   IMAD.HI.X R21, R42, 0x8, R9;                                /* 0x2092c00022a55ce3 */
        /*0778*/                   MOV32I R16, 0x1;                                            /* 0x1800000004041de2 */
                                                                                               /* 0x23f2929280429047 */
        /*0788*/                   LD.E.64 R10, [R20+0x18];                                    /* 0x8400000061429ca5 */
        /*0790*/                   DFMA R12, R28, R28, c[0x2][0x18];                           /* 0x2038880061c31c01 */
        /*0798*/                   MUFU.RCP64H R17, R13;                                       /* 0xc800000018d45c00 */
        /*07a0*/                   SSY 0x858;                                                  /* 0x60000002c0001c07 */
        /*07a8*/                   DFMA R18, -R12, R16, c[0x2][0x0];                           /* 0x2020880000c49e01 */
        /*07b0*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*07b8*/                   DFMA R16, R16, R18, R16;                                    /* 0x2020000049041c01 */
                                                                                               /* 0x2282820292929297 */
        /*07c8*/                   DMUL R10, R10, c[0x2][0x10];                                /* 0x5000480040a29c01 */
        /*07d0*/                   DMUL R18, R10, R14;                                         /* 0x5000000038a49c01 */
        /*07d8*/                   DMUL R30, R18, R16;                                         /* 0x5000000041279c01 */
        /*07e0*/                   DFMA R10, -R12, R30, R18;                                   /* 0x2024000078c29e01 */
        /*07e8*/                   DFMA R30, R16, R10, R30;                                    /* 0x203c000029079c01 */
        /*07f0*/                   FSET.GTU.AND RZ.CC, |R19|, c[0x2][0xc], PT;                 /* 0x160f4800313fdc80 */
        /*07f8*/                   FFMA R9, RZ, R13, R31;                                      /* 0x303e000037f25c00 */
                                                                                               /* 0x2202b2b20042b2c7 */
        /*0808*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*0810*/               @P0 BRA CC.NEU, 0x850;                                          /* 0x40000000e00001a7 */
        /*0818*/                   MOV R10, R18;                                               /* 0x2800000048029de4 */
        /*0820*/                   MOV R11, R19;                                               /* 0x280000004c02dde4 */
        /*0828*/                   MOV R9, R13;                                                /* 0x2800000034025de4 */
        /*0830*/                   CAL 0x17e0;                                                 /* 0x5000003ea0011c07 */
        /*0838*/                   MOV R30, R9;                                                /* 0x2800000024079de4 */
                                                                                               /* 0x229292928042b2b7 */
        /*0848*/                   MOV R31, R10;                                               /* 0x280000002807dde4 */
        /*0850*/                   MUFU.RCP64H.S R11, R29;                                     /* 0xc800000019d2dc10 */
        /*0858*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*0860*/                   DMUL R16, R26, R30;                                         /* 0x5000000079a41c01 */
        /*0868*/                   DFMA R12, -R28, R10, c[0x2][0x0];                           /* 0x2014880001c31e01 */
        /*0870*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*0878*/                   DFMA R10, R10, R12, R10;                                    /* 0x2014000030a29c01 */
                                                                                               /* 0x22c2828042029297 */
        /*0888*/                   DMUL R14, R16, R10;                                         /* 0x5000000029039c01 */
        /*0890*/                   DFMA R12, -R28, R14, R16;                                   /* 0x2020000039c31e01 */
        /*0898*/                   DFMA R10, R10, R12, R14;                                    /* 0x201c000030a29c01 */
        /*08a0*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0xc], PT;                 /* 0x160f4800311fdc80 */
        /*08a8*/                   SSY 0x918;                                                  /* 0x60000001a0001c07 */
        /*08b0*/                   FFMA R9, RZ, R29, R11;                                      /* 0x3016000077f25c00 */
        /*08b8*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
                                                                                               /* 0x2202b2b0420042b7 */
        /*08c8*/               @P0 BRA CC.NEU, 0x910;                                          /* 0x40000001000001a7 */
        /*08d0*/                   MOV R10, R16;                                               /* 0x2800000040029de4 */
        /*08d8*/                   MOV R11, R17;                                               /* 0x280000004402dde4 */
        /*08e0*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*08e8*/                   MOV R9, R29;                                                /* 0x2800000074025de4 */
        /*08f0*/                   CAL 0x17e0;                                                 /* 0x5000003ba0011c07 */
        /*08f8*/                   MOV R11, R10;                                               /* 0x280000002802dde4 */
                                                                                               /* 0x228200423042b2b7 */
        /*0908*/                   MOV R10, R9;                                                /* 0x2800000024029de4 */
        /*0910*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0918*/                   LD.E.64 R12, [R2+0x4e218];                                  /* 0x8400138860231ca5 */
        /*0920*/                   MUFU.RCP64H R15, R29;                                       /* 0xc800000019d3dc00 */
        /*0928*/                   LD.E.64 R26, [R2+0x5dc18];                                  /* 0x8400177060269ca5 */
        /*0930*/                   DMUL R24, R24, R30;                                         /* 0x5000000079861c01 */
        /*0938*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
                                                                                               /* 0x2282029292929297 */
        /*0948*/                   DFMA R16, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c41e01 */
        /*0950*/                   DFMA R16, R16, R16, R16;                                    /* 0x2020000041041c01 */
        /*0958*/                   DFMA R18, R14, R16, R14;                                    /* 0x201c000040e49c01 */
        /*0960*/                   DMUL R14, R18, R24;                                         /* 0x5000000061239c01 */
        /*0968*/                   DFMA R16, -R28, R14, R24;                                   /* 0x2030000039c41e01 */
        /*0970*/                   DFMA R16, R18, R16, R14;                                    /* 0x201c000041241c01 */
        /*0978*/                   FSET.GTU.AND RZ.CC, |R25|, c[0x2][0xc], PT;                 /* 0x160f4800319fdc80 */
                                                                                               /* 0x2202b04293f28047 */
        /*0988*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*0990*/                   SSY 0xa00;                                                  /* 0x60000001a0001c07 */
        /*0998*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*09a0*/                   DADD R12, R10, R12;                                         /* 0x4800000030a31c01 */
        /*09a8*/                   ST.E.64 [R2+0x4e218], R12;                                  /* 0x9400138860231ca5 */
        /*09b0*/               @P0 BRA CC.NEU, 0x9f8;                                          /* 0x40000001000001a7 */
        /*09b8*/                   MOV R10, R24;                                               /* 0x2800000060029de4 */
                                                                                               /* 0x22b2b042b2b20047 */
        /*09c8*/                   MOV R11, R25;                                               /* 0x280000006402dde4 */
        /*09d0*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*09d8*/                   MOV R9, R29;                                                /* 0x2800000074025de4 */
        /*09e0*/                   CAL 0x17e0;                                                 /* 0x50000037e0011c07 */
        /*09e8*/                   MOV R16, R9;                                                /* 0x2800000024041de4 */
        /*09f0*/                   MOV R17, R10;                                               /* 0x2800000028045de4 */
        /*09f8*/                   DADD.S R16, R26, R16;                                       /* 0x4800000041a41c11 */
                                                                                               /* 0x228042b2c2c042b7 */
        /*0a08*/                   ST.E.64.S [R2+0x5dc18], R16;                                /* 0x9400177060241cb5 */
        /*0a10*/                   ISUB RZ.CC, R6, R33;                                        /* 0x48010000846fdd03 */
        /*0a18*/                   SSY 0xe70;                                                  /* 0x6000001140001c07 */
        /*0a20*/                   ISETP.EQ.X.AND P0, PT, R39, R36, PT;                        /* 0x190e00009271dc63 */
        /*0a28*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*0a30*/                   MOV32I R9, 0x3;                                             /* 0x180000000c025de2 */
        /*0a38*/                   SHL R10, R6, -0x1d;                                         /* 0x6000ffff8c629c03 */
                                                                                               /* 0x2202804220420047 */
        /*0a48*/                   IADD32I RZ.CC, R9, -0x20;                                   /* 0x0fffffff809fdc02 */
        /*0a50*/                   SHL R11, R39, 0x3;                                          /* 0x6000c0000e72dc03 */
        /*0a58*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
        /*0a60*/                   LD.E.64 R24, [R2+0x1f418];                                  /* 0x840007d060261ca5 */
        /*0a68*/                   SHR.U32 R12, R6, 0x1d;                                      /* 0x5800c00074631c03 */
        /*0a70*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*0a78*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
                                                                                               /* 0x228270420042b047 */
        /*0a88*/                   IMAD.U32.U32 R14.CC, R6, 0x8, R18;                          /* 0x2025c00020639c03 */
        /*0a90*/                   LD.E.64 R26, [R2+0xfa18];                                   /* 0x840003e860269ca5 */
        /*0a98*/               @P0 IADD R10, R11, R12;                                         /* 0x4800000030b28003 */
        /*0aa0*/                   MOV32I R20, 0x0;                                            /* 0x1800000000051de2 */
        /*0aa8*/                   MOV R12, R14;                                               /* 0x2800000038031de4 */
        /*0ab0*/                   MOV32I R21, 0x3fe00000;                                     /* 0x18ff800000055de2 */
        /*0ab8*/                   IADD.X R13, R10, R9;                                        /* 0x4800000024a35c43 */
                                                                                               /* 0x2292925233f23047 */
        /*0ac8*/                   LD.E.64 R10, [R12+0x1f410];                                 /* 0x840007d040c29ca5 */
        /*0ad0*/                   SSY 0xbc0;                                                  /* 0x60000003a0001c07 */
        /*0ad8*/                   LD.E.64 R14, [R12+0xfa10];                                  /* 0x840003e840c39ca5 */
        /*0ae0*/                   DADD R24, -R24, R10;                                        /* 0x4800000029861e01 */
        /*0ae8*/                   DADD R26, -R26, R14;                                        /* 0x4800000039a69e01 */
        /*0af0*/                   DMUL R14, R24, R24;                                         /* 0x5000000061839c01 */
        /*0af8*/                   DFMA R10, R26, R26, R14;                                    /* 0x201c000069a29c01 */
                                                                                               /* 0x2292329292828047 */
        /*0b08*/                   IADD32I R19, R11, -0x3500000;                               /* 0x0bf2c00000b4dc02 */
        /*0b10*/                   MUFU.RSQ64H R15, R11;                                       /* 0xc80000001cb3dc00 */
        /*0b18*/                   MOV R14, R19;                                               /* 0x280000004c039de4 */
        /*0b20*/                   DMUL R16, R14, R14;                                         /* 0x5000000038e41c01 */
        /*0b28*/                   DFMA R16, -R10, R16, c[0x2][0x0];                           /* 0x2020880000a41e01 */
        /*0b30*/                   DFMA R12, R16, 0.375, R20;                                  /* 0x2028cff601031c01 */
        /*0b38*/                   DMUL R16, R14, R16;                                         /* 0x5000000040e41c01 */
                                                                                               /* 0x2282028042028207 */
        /*0b48*/                   DFMA R20, R12, R16, R14;                                    /* 0x201c000040c51c01 */
        /*0b50*/                   ISETP.LT.U32.AND P0, PT, R19, c[0x2][0x8], PT;              /* 0x188e48002131dc03 */
        /*0b58*/                   DMUL R16, R10, R20;                                         /* 0x5000000050a41c01 */
        /*0b60*/                   IADD32I R23, R21, -0x100000;                                /* 0x0bffc0000155dc02 */
        /*0b68*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*0b70*/                   DFMA R12, -R16, R16, R10;                                   /* 0x2014000041031e01 */
        /*0b78*/                   MOV R15, R23;                                               /* 0x280000005c03dde4 */
                                                                                               /* 0x22b2b2b20042b207 */
        /*0b88*/                   DFMA R28, R12, R14, R16;                                    /* 0x2020000038c71c01 */
        /*0b90*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*0b98*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*0ba0*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*0ba8*/                   MOV R19, R23;                                               /* 0x280000005c04dde4 */
        /*0bb0*/                   CAL 0x1b60;                                                 /* 0x5000003ea0011c07 */
        /*0bb8*/                   NOP.S;                                                      /* 0x4000000000001df4 */
                                                                                               /* 0x228290428042c047 */
        /*0bc8*/                   IMAD R20.CC, R42, 0x8, R18;                                 /* 0x2025c00022a51ca3 */
        /*0bd0*/                   LD.E.64 R14, [R4+0x28];                                     /* 0x84000000a0439ca5 */
        /*0bd8*/                   IMAD.HI.X R21, R42, 0x8, R9;                                /* 0x2092c00022a55ce3 */
        /*0be0*/                   MOV32I R16, 0x1;                                            /* 0x1800000004041de2 */
        /*0be8*/                   LD.E.64 R10, [R20+0x18];                                    /* 0x8400000061429ca5 */
        /*0bf0*/                   DFMA R12, R28, R28, c[0x2][0x18];                           /* 0x2038880061c31c01 */
        /*0bf8*/                   MUFU.RCP64H R17, R13;                                       /* 0xc800000018d45c00 */
                                                                                               /* 0x22929293f2029297 */
        /*0c08*/                   DFMA R18, -R12, R16, c[0x2][0x0];                           /* 0x2020880000c49e01 */
        /*0c10*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*0c18*/                   DFMA R16, R16, R18, R16;                                    /* 0x2020000049041c01 */
        /*0c20*/                   SSY 0xcb8;                                                  /* 0x6000000240001c07 */
        /*0c28*/                   DMUL R10, R10, c[0x2][0x10];                                /* 0x5000480040a29c01 */
        /*0c30*/                   DMUL R18, R10, R14;                                         /* 0x5000000038a49c01 */
        /*0c38*/                   DMUL R30, R18, R16;                                         /* 0x5000000041279c01 */
                                                                                               /* 0x2202b2c282820297 */
        /*0c48*/                   DFMA R10, -R12, R30, R18;                                   /* 0x2024000078c29e01 */
        /*0c50*/                   DFMA R30, R16, R10, R30;                                    /* 0x203c000029079c01 */
        /*0c58*/                   FSET.GTU.AND RZ.CC, |R19|, c[0x2][0xc], PT;                 /* 0x160f4800313fdc80 */
        /*0c60*/                   FFMA R9, RZ, R13, R31;                                      /* 0x303e000037f25c00 */
        /*0c68*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*0c70*/               @P0 BRA CC.NEU, 0xcb0;                                          /* 0x40000000e00001a7 */
        /*0c78*/                   MOV R10, R18;                                               /* 0x2800000048029de4 */
                                                                                               /* 0x2282b2b042b2b047 */
        /*0c88*/                   MOV R11, R19;                                               /* 0x280000004c02dde4 */
        /*0c90*/                   MOV R9, R13;                                                /* 0x2800000034025de4 */
        /*0c98*/                   CAL 0x17e0;                                                 /* 0x5000002d00011c07 */
        /*0ca0*/                   MOV R30, R9;                                                /* 0x2800000024079de4 */
        /*0ca8*/                   MOV R31, R10;                                               /* 0x280000002807dde4 */
        /*0cb0*/                   MUFU.RCP64H.S R11, R29;                                     /* 0xc800000019d2dc10 */
        /*0cb8*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
                                                                                               /* 0x2202929292523297 */
        /*0cc8*/                   DFMA R12, -R28, R10, c[0x2][0x0];                           /* 0x2014880001c31e01 */
        /*0cd0*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*0cd8*/                   DMUL R16, R26, R30;                                         /* 0x5000000079a41c01 */
        /*0ce0*/                   DFMA R10, R10, R12, R10;                                    /* 0x2014000030a29c01 */
        /*0ce8*/                   DMUL R14, R16, R10;                                         /* 0x5000000029039c01 */
        /*0cf0*/                   DFMA R12, -R28, R14, R16;                                   /* 0x2020000039c31e01 */
        /*0cf8*/                   DFMA R10, R10, R12, R14;                                    /* 0x201c000030a29c01 */
                                                                                               /* 0x220042b2c2828047 */
        /*0d08*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0xc], PT;                 /* 0x160f4800311fdc80 */
        /*0d10*/                   SSY 0xd78;                                                  /* 0x6000000180001c07 */
        /*0d18*/                   FFMA R9, RZ, R29, R11;                                      /* 0x3016000077f25c00 */
        /*0d20*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*0d28*/               @P0 BRA CC.NEU, 0xd70;                                          /* 0x40000001000001a7 */
        /*0d30*/                   MOV R10, R16;                                               /* 0x2800000040029de4 */
        /*0d38*/                   MOV R11, R17;                                               /* 0x280000004402dde4 */
                                                                                               /* 0x2232b2b042b2b047 */
        /*0d48*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*0d50*/                   MOV R9, R29;                                                /* 0x2800000074025de4 */
        /*0d58*/                   CAL 0x17e0;                                                 /* 0x5000002a00011c07 */
        /*0d60*/                   MOV R11, R10;                                               /* 0x280000002802dde4 */
        /*0d68*/                   MOV R10, R9;                                                /* 0x2800000024029de4 */
        /*0d70*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0d78*/                   LD.E.64 R12, [R2+0x4e218];                                  /* 0x8400138860231ca5 */
                                                                                               /* 0x2292929280420047 */
        /*0d88*/                   LD.E.64 R26, [R2+0x5dc18];                                  /* 0x8400177060269ca5 */
        /*0d90*/                   MUFU.RCP64H R15, R29;                                       /* 0xc800000019d3dc00 */
        /*0d98*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*0da0*/                   DMUL R24, R24, R30;                                         /* 0x5000000079861c01 */
        /*0da8*/                   DFMA R16, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c41e01 */
        /*0db0*/                   DFMA R16, R16, R16, R16;                                    /* 0x2020000041041c01 */
        /*0db8*/                   DFMA R18, R14, R16, R14;                                    /* 0x201c000040e49c01 */
                                                                                               /* 0x23f2828042029297 */
        /*0dc8*/                   DMUL R14, R18, R24;                                         /* 0x5000000061239c01 */
        /*0dd0*/                   DFMA R16, -R28, R14, R24;                                   /* 0x2030000039c41e01 */
        /*0dd8*/                   DFMA R16, R18, R16, R14;                                    /* 0x201c000041241c01 */
        /*0de0*/                   FSET.GTU.AND RZ.CC, |R25|, c[0x2][0xc], PT;                 /* 0x160f4800319fdc80 */
        /*0de8*/                   SSY 0xe68;                                                  /* 0x60000001e0001c07 */
        /*0df0*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*0df8*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
                                                                                               /* 0x22b0420042b04297 */
        /*0e08*/                   DADD R12, R10, R12;                                         /* 0x4800000030a31c01 */
        /*0e10*/                   ST.E.64 [R2+0x4e218], R12;                                  /* 0x9400138860231ca5 */
        /*0e18*/               @P0 BRA CC.NEU, 0xe60;                                          /* 0x40000001000001a7 */
        /*0e20*/                   MOV R10, R24;                                               /* 0x2800000060029de4 */
        /*0e28*/                   MOV R11, R25;                                               /* 0x280000006402dde4 */
        /*0e30*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*0e38*/                   MOV R9, R29;                                                /* 0x2800000074025de4 */
                                                                                               /* 0x22c042b2b2b042b7 */
        /*0e48*/                   CAL 0x17e0;                                                 /* 0x5000002640011c07 */
        /*0e50*/                   MOV R16, R9;                                                /* 0x2800000024041de4 */
        /*0e58*/                   MOV R17, R10;                                               /* 0x2800000028045de4 */
        /*0e60*/                   DADD.S R16, R26, R16;                                       /* 0x4800000041a41c11 */
        /*0e68*/                   ST.E.64.S [R2+0x5dc18], R16;                                /* 0x9400177060241cb5 */
        /*0e70*/                   ISUB RZ.CC, R7, R33;                                        /* 0x48010000847fdd03 */
        /*0e78*/                   SSY 0x12d8;                                                 /* 0x6000001160001c07 */
                                                                                               /* 0x222200428042b2c7 */
        /*0e88*/                   ISETP.EQ.X.AND P0, PT, R38, R36, PT;                        /* 0x190e00009261dc63 */
        /*0e90*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*0e98*/                   MOV32I R9, 0x3;                                             /* 0x180000000c025de2 */
        /*0ea0*/                   SHL R10, R7, -0x1d;                                         /* 0x6000ffff8c729c03 */
        /*0ea8*/                   IADD32I RZ.CC, R9, -0x20;                                   /* 0x0fffffff809fdc02 */
        /*0eb0*/                   SHL R11, R38, 0x3;                                          /* 0x6000c0000e62dc03 */
        /*0eb8*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
                                                                                               /* 0x2202b04200428047 */
        /*0ec8*/                   SHR.U32 R12, R7, 0x1d;                                      /* 0x5800c00074731c03 */
        /*0ed0*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*0ed8*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*0ee0*/                   LD.E.64 R24, [R2+0x1f418];                                  /* 0x840007d060261ca5 */
        /*0ee8*/                   IMAD.U32.U32 R14.CC, R7, 0x8, R18;                          /* 0x2025c00020739c03 */
        /*0ef0*/                   MOV32I R20, 0x0;                                            /* 0x1800000000051de2 */
        /*0ef8*/               @P0 IADD R10, R11, R12;                                         /* 0x4800000030b28003 */
                                                                                               /* 0x23f2304280427047 */
        /*0f08*/                   MOV R12, R14;                                               /* 0x2800000038031de4 */
        /*0f10*/                   LD.E.64 R26, [R2+0xfa18];                                   /* 0x840003e860269ca5 */
        /*0f18*/                   IADD.X R13, R10, R9;                                        /* 0x4800000024a35c43 */
        /*0f20*/                   MOV32I R21, 0x3fe00000;                                     /* 0x18ff800000055de2 */
        /*0f28*/                   LD.E.64 R10, [R12+0x1f410];                                 /* 0x840007d040c29ca5 */
        /*0f30*/                   SSY 0x1028;                                                 /* 0x60000003c0001c07 */
        /*0f38*/                   LD.E.64 R14, [R12+0xfa10];                                  /* 0x840003e840c39ca5 */
                                                                                               /* 0x2282804292925237 */
        /*0f48*/                   DADD R24, -R24, R10;                                        /* 0x4800000029861e01 */
        /*0f50*/                   DADD R26, -R26, R14;                                        /* 0x4800000039a69e01 */
        /*0f58*/                   DMUL R14, R24, R24;                                         /* 0x5000000061839c01 */
        /*0f60*/                   DFMA R10, R26, R26, R14;                                    /* 0x201c000069a29c01 */
        /*0f68*/                   IADD32I R19, R11, -0x3500000;                               /* 0x0bf2c00000b4dc02 */
        /*0f70*/                   MUFU.RSQ64H R15, R11;                                       /* 0xc80000001cb3dc00 */
        /*0f78*/                   MOV R14, R19;                                               /* 0x280000004c039de4 */
                                                                                               /* 0x2202820292329297 */
        /*0f88*/                   DMUL R16, R14, R14;                                         /* 0x5000000038e41c01 */
        /*0f90*/                   DFMA R16, -R10, R16, c[0x2][0x0];                           /* 0x2020880000a41e01 */
        /*0f98*/                   DFMA R12, R16, 0.375, R20;                                  /* 0x2028cff601031c01 */
        /*0fa0*/                   DMUL R16, R14, R16;                                         /* 0x5000000040e41c01 */
        /*0fa8*/                   DFMA R20, R12, R16, R14;                                    /* 0x201c000040c51c01 */
        /*0fb0*/                   ISETP.LT.U32.AND P0, PT, R19, c[0x2][0x8], PT;              /* 0x188e48002131dc03 */
        /*0fb8*/                   DMUL R16, R10, R20;                                         /* 0x5000000050a41c01 */
                                                                                               /* 0x2202b20282028047 */
        /*0fc8*/                   IADD32I R23, R21, -0x100000;                                /* 0x0bffc0000155dc02 */
        /*0fd0*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*0fd8*/                   DFMA R12, -R16, R16, R10;                                   /* 0x2014000041031e01 */
        /*0fe0*/                   MOV R15, R23;                                               /* 0x280000005c03dde4 */
        /*0fe8*/                   DFMA R28, R12, R14, R16;                                    /* 0x2020000038c71c01 */
        /*0ff0*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*0ff8*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
                                                                                               /* 0x2282c042b2b2b047 */
        /*1008*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*1010*/                   MOV R19, R23;                                               /* 0x280000005c04dde4 */
        /*1018*/                   CAL 0x1b60;                                                 /* 0x5000002d00011c07 */
        /*1020*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*1028*/                   IMAD R20.CC, R42, 0x8, R18;                                 /* 0x2025c00022a51ca3 */
        /*1030*/                   LD.E.64 R14, [R4+0x30];                                     /* 0x84000000c0439ca5 */
        /*1038*/                   IMAD.HI.X R21, R42, 0x8, R9;                                /* 0x2092c00022a55ce3 */
                                                                                               /* 0x2292928280420047 */
        /*1048*/                   LD.E.64 R10, [R20+0x18];                                    /* 0x8400000061429ca5 */
        /*1050*/                   DFMA R12, R28, R28, c[0x2][0x18];                           /* 0x2038880061c31c01 */
        /*1058*/                   MOV32I R16, 0x1;                                            /* 0x1800000004041de2 */
        /*1060*/                   SSY 0x1120;                                                 /* 0x60000002e0001c07 */
        /*1068*/                   MUFU.RCP64H R17, R13;                                       /* 0xc800000018d45c00 */
        /*1070*/                   DFMA R18, -R12, R16, c[0x2][0x0];                           /* 0x2020880000c49e01 */
        /*1078*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
                                                                                               /* 0x22820292929293f7 */
        /*1088*/                   DFMA R16, R16, R18, R16;                                    /* 0x2020000049041c01 */
        /*1090*/                   DMUL R10, R10, c[0x2][0x10];                                /* 0x5000480040a29c01 */
        /*1098*/                   DMUL R18, R10, R14;                                         /* 0x5000000038a49c01 */
        /*10a0*/                   DMUL R30, R18, R16;                                         /* 0x5000000041279c01 */
        /*10a8*/                   DFMA R10, -R12, R30, R18;                                   /* 0x2024000078c29e01 */
        /*10b0*/                   DFMA R30, R16, R10, R30;                                    /* 0x203c000029079c01 */
        /*10b8*/                   FSET.GTU.AND RZ.CC, |R19|, c[0x2][0xc], PT;                 /* 0x160f4800313fdc80 */
                                                                                               /* 0x22b2b20042b2c287 */
        /*10c8*/                   FFMA R9, RZ, R13, R31;                                      /* 0x303e000037f25c00 */
        /*10d0*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*10d8*/               @P0 BRA CC.NEU, 0x1118;                                         /* 0x40000000e00001a7 */
        /*10e0*/                   MOV R10, R18;                                               /* 0x2800000048029de4 */
        /*10e8*/                   MOV R11, R19;                                               /* 0x280000004c02dde4 */
        /*10f0*/                   MOV R9, R13;                                                /* 0x2800000034025de4 */
        /*10f8*/                   CAL 0x17e0;                                                 /* 0x5000001b80011c07 */
                                                                                               /* 0x2292928042b2b047 */
        /*1108*/                   MOV R30, R9;                                                /* 0x2800000024079de4 */
        /*1110*/                   MOV R31, R10;                                               /* 0x280000002807dde4 */
        /*1118*/                   MUFU.RCP64H.S R11, R29;                                     /* 0xc800000019d2dc10 */
        /*1120*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*1128*/                   DMUL R16, R26, R30;                                         /* 0x5000000079a41c01 */
        /*1130*/                   DFMA R12, -R28, R10, c[0x2][0x0];                           /* 0x2014880001c31e01 */
        /*1138*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
                                                                                               /* 0x2282804202929297 */
        /*1148*/                   DFMA R10, R10, R12, R10;                                    /* 0x2014000030a29c01 */
        /*1150*/                   DMUL R14, R16, R10;                                         /* 0x5000000029039c01 */
        /*1158*/                   DFMA R12, -R28, R14, R16;                                   /* 0x2020000039c31e01 */
        /*1160*/                   DFMA R10, R10, R12, R14;                                    /* 0x201c000030a29c01 */
        /*1168*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0xc], PT;                 /* 0x160f4800311fdc80 */
        /*1170*/                   SSY 0x11e0;                                                 /* 0x60000001a0001c07 */
        /*1178*/                   FFMA R9, RZ, R29, R11;                                      /* 0x3016000077f25c00 */
                                                                                               /* 0x22b2b0420042b2c7 */
        /*1188*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*1190*/               @P0 BRA CC.NEU, 0x11d8;                                         /* 0x40000001000001a7 */
        /*1198*/                   MOV R10, R16;                                               /* 0x2800000040029de4 */
        /*11a0*/                   MOV R11, R17;                                               /* 0x280000004402dde4 */
        /*11a8*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*11b0*/                   MOV R9, R29;                                                /* 0x2800000074025de4 */
        /*11b8*/                   CAL 0x17e0;                                                 /* 0x5000001880011c07 */
                                                                                               /* 0x2200423042b2b047 */
        /*11c8*/                   MOV R11, R10;                                               /* 0x280000002802dde4 */
        /*11d0*/                   MOV R10, R9;                                                /* 0x2800000024029de4 */
        /*11d8*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*11e0*/                   LD.E.64 R12, [R2+0x4e218];                                  /* 0x8400138860231ca5 */
        /*11e8*/                   MUFU.RCP64H R15, R29;                                       /* 0xc800000019d3dc00 */
        /*11f0*/                   LD.E.64 R26, [R2+0x5dc18];                                  /* 0x8400177060269ca5 */
        /*11f8*/                   DMUL R24, R24, R30;                                         /* 0x5000000079861c01 */
                                                                                               /* 0x2292929292928047 */
        /*1208*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*1210*/                   SSY 0x12d0;                                                 /* 0x60000002e0001c07 */
        /*1218*/                   DFMA R16, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c41e01 */
        /*1220*/                   DFMA R16, R16, R16, R16;                                    /* 0x2020000041041c01 */
        /*1228*/                   DFMA R18, R14, R16, R14;                                    /* 0x201c000040e49c01 */
        /*1230*/                   DMUL R14, R18, R24;                                         /* 0x5000000061239c01 */
        /*1238*/                   DFMA R16, -R28, R14, R24;                                   /* 0x2030000039c41e01 */
                                                                                               /* 0x22b04293f2828207 */
        /*1248*/                   DFMA R16, R18, R16, R14;                                    /* 0x201c000041241c01 */
        /*1250*/                   FSET.GTU.AND RZ.CC, |R25|, c[0x2][0xc], PT;                 /* 0x160f4800319fdc80 */
        /*1258*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*1260*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*1268*/                   DADD R12, R10, R12;                                         /* 0x4800000030a31c01 */
        /*1270*/                   ST.E.64 [R2+0x4e218], R12;                                  /* 0x9400138860231ca5 */
        /*1278*/               @P0 BRA CC.NEU, 0x12c0;                                         /* 0x40000001000001a7 */
                                                                                               /* 0x22b042b2b0420047 */
        /*1288*/                   MOV R10, R24;                                               /* 0x2800000060029de4 */
        /*1290*/                   MOV R11, R25;                                               /* 0x280000006402dde4 */
        /*1298*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*12a0*/                   MOV R9, R29;                                                /* 0x2800000074025de4 */
        /*12a8*/                   CAL 0x17e0;                                                 /* 0x50000014c0011c07 */
        /*12b0*/                   MOV R16, R9;                                                /* 0x2800000024041de4 */
        /*12b8*/                   MOV R17, R10;                                               /* 0x2800000028045de4 */
                                                                                               /* 0x2282b2c2c042b2b7 */
        /*12c8*/                   DADD.S R16, R26, R16;                                       /* 0x4800000041a41c11 */
        /*12d0*/                   ST.E.64.S [R2+0x5dc18], R16;                                /* 0x9400177060241cb5 */
        /*12d8*/                   ISUB RZ.CC, R8, R33;                                        /* 0x48010000848fdd03 */
        /*12e0*/                   SSY 0x1738;                                                 /* 0x6000001140001c07 */
        /*12e8*/                   ISETP.EQ.X.AND P0, PT, R37, R36, PT;                        /* 0x190e00009251dc63 */
        /*12f0*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*12f8*/                   MOV32I R9, 0x3;                                             /* 0x180000000c025de2 */
                                                                                               /* 0x2242304220420047 */
        /*1308*/                   IADD32I RZ.CC, R9, -0x20;                                   /* 0x0fffffff809fdc02 */
        /*1310*/                   SHL R10, R8, -0x1d;                                         /* 0x6000ffff8c829c03 */
        /*1318*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
        /*1320*/                   LD.E.64 R24, [R2+0x1f418];                                  /* 0x840007d060261ca5 */
        /*1328*/                   SHL R11, R37, 0x3;                                          /* 0x6000c0000e52dc03 */
        /*1330*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*1338*/                   SHR.U32 R12, R8, 0x1d;                                      /* 0x5800c00074831c03 */
                                                                                               /* 0x22720042b0420047 */
        /*1348*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*1350*/                   LD.E.64 R26, [R2+0xfa18];                                   /* 0x840003e860269ca5 */
        /*1358*/                   IMAD.U32.U32 R14.CC, R8, 0x8, R18;                          /* 0x2025c00020839c03 */
        /*1360*/                   MOV32I R20, 0x0;                                            /* 0x1800000000051de2 */
        /*1368*/               @P0 IADD R10, R11, R12;                                         /* 0x4800000030b28003 */
        /*1370*/                   MOV32I R21, 0x3fe00000;                                     /* 0x18ff800000055de2 */
        /*1378*/                   MOV R12, R14;                                               /* 0x2800000038031de4 */
                                                                                               /* 0x22925233f2328047 */
        /*1388*/                   IADD.X R13, R10, R9;                                        /* 0x4800000024a35c43 */
        /*1390*/                   SSY 0x1490;                                                 /* 0x60000003e0001c07 */
        /*1398*/                   LD.E.64 R10, [R12+0x1f410];                                 /* 0x840007d040c29ca5 */
        /*13a0*/                   LD.E.64 R14, [R12+0xfa10];                                  /* 0x840003e840c39ca5 */
        /*13a8*/                   DADD R24, -R24, R10;                                        /* 0x4800000029861e01 */
        /*13b0*/                   DADD R26, -R26, R14;                                        /* 0x4800000039a69e01 */
        /*13b8*/                   DMUL R14, R24, R24;                                         /* 0x5000000061839c01 */
                                                                                               /* 0x2232929282804297 */
        /*13c8*/                   DFMA R10, R26, R26, R14;                                    /* 0x201c000069a29c01 */
        /*13d0*/                   IADD32I R19, R11, -0x3500000;                               /* 0x0bf2c00000b4dc02 */
        /*13d8*/                   MUFU.RSQ64H R15, R11;                                       /* 0xc80000001cb3dc00 */
        /*13e0*/                   MOV R14, R19;                                               /* 0x280000004c039de4 */
        /*13e8*/                   DMUL R16, R14, R14;                                         /* 0x5000000038e41c01 */
        /*13f0*/                   DFMA R16, -R10, R16, c[0x2][0x0];                           /* 0x2020880000a41e01 */
        /*13f8*/                   DFMA R12, R16, 0.375, R20;                                  /* 0x2028cff601031c01 */
                                                                                               /* 0x2202804202820297 */
        /*1408*/                   DMUL R16, R14, R16;                                         /* 0x5000000040e41c01 */
        /*1410*/                   DFMA R20, R12, R16, R14;                                    /* 0x201c000040c51c01 */
        /*1418*/                   ISETP.LT.U32.AND P0, PT, R19, c[0x2][0x8], PT;              /* 0x188e48002131dc03 */
        /*1420*/                   DMUL R16, R10, R20;                                         /* 0x5000000050a41c01 */
        /*1428*/                   IADD32I R23, R21, -0x100000;                                /* 0x0bffc0000155dc02 */
        /*1430*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*1438*/                   DFMA R12, -R16, R16, R10;                                   /* 0x2014000041031e01 */
                                                                                               /* 0x22b2b20042b20287 */
        /*1448*/                   MOV R15, R23;                                               /* 0x280000005c03dde4 */
        /*1450*/                   DFMA R28, R12, R14, R16;                                    /* 0x2020000038c71c01 */
        /*1458*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*1460*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*1468*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*1470*/                   MOV R19, R23;                                               /* 0x280000005c04dde4 */
        /*1478*/                   CAL 0x1b60;                                                 /* 0x5000001b80011c07 */
                                                                                               /* 0x2290428042c042b7 */
        /*1488*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*1490*/                   IMAD R20.CC, R42, 0x8, R18;                                 /* 0x2025c00022a51ca3 */
        /*1498*/                   LD.E.64 R14, [R4+0x38];                                     /* 0x84000000e0439ca5 */
        /*14a0*/                   IMAD.HI.X R21, R42, 0x8, R9;                                /* 0x2092c00022a55ce3 */
        /*14a8*/                   MOV32I R16, 0x1;                                            /* 0x1800000004041de2 */
        /*14b0*/                   LD.E.64 R10, [R20+0x18];                                    /* 0x8400000061429ca5 */
        /*14b8*/                   DFMA R12, R28, R28, c[0x2][0x18];                           /* 0x2038880061c31c01 */
                                                                                               /* 0x229293f292928047 */
        /*14c8*/                   MUFU.RCP64H R17, R13;                                       /* 0xc800000018d45c00 */
        /*14d0*/                   SSY 0x1580;                                                 /* 0x60000002a0001c07 */
        /*14d8*/                   DFMA R18, -R12, R16, c[0x2][0x0];                           /* 0x2020880000c49e01 */
        /*14e0*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*14e8*/                   DFMA R16, R16, R18, R16;                                    /* 0x2020000049041c01 */
        /*14f0*/                   DMUL R10, R10, c[0x2][0x10];                                /* 0x5000480040a29c01 */
        /*14f8*/                   DMUL R18, R10, R14;                                         /* 0x5000000038a49c01 */
                                                                                               /* 0x22b2c28282029297 */
        /*1508*/                   DMUL R30, R18, R16;                                         /* 0x5000000041279c01 */
        /*1510*/                   DFMA R10, -R12, R30, R18;                                   /* 0x2024000078c29e01 */
        /*1518*/                   DFMA R30, R16, R10, R30;                                    /* 0x203c000029079c01 */
        /*1520*/                   FSET.GTU.AND RZ.CC, |R19|, c[0x2][0xc], PT;                 /* 0x160f4800313fdc80 */
        /*1528*/                   FFMA R9, RZ, R13, R31;                                      /* 0x303e000037f25c00 */
        /*1530*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*1538*/               @P0 BRA CC.NEU, 0x1578;                                         /* 0x40000000e00001a7 */
                                                                                               /* 0x22b2b042b2b20047 */
        /*1548*/                   MOV R10, R18;                                               /* 0x2800000048029de4 */
        /*1550*/                   MOV R11, R19;                                               /* 0x280000004c02dde4 */
        /*1558*/                   MOV R9, R13;                                                /* 0x2800000034025de4 */
        /*1560*/                   CAL 0x17e0;                                                 /* 0x50000009e0011c07 */
        /*1568*/                   MOV R30, R9;                                                /* 0x2800000024079de4 */
        /*1570*/                   MOV R31, R10;                                               /* 0x280000002807dde4 */
        /*1578*/                   MUFU.RCP64H.S R11, R29;                                     /* 0xc800000019d2dc10 */
                                                                                               /* 0x2292929292928047 */
        /*1588*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*1590*/                   DMUL R16, R26, R30;                                         /* 0x5000000079a41c01 */
        /*1598*/                   DFMA R12, -R28, R10, c[0x2][0x0];                           /* 0x2014880001c31e01 */
        /*15a0*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*15a8*/                   DFMA R10, R10, R12, R10;                                    /* 0x2014000030a29c01 */
        /*15b0*/                   DMUL R14, R16, R10;                                         /* 0x5000000029039c01 */
        /*15b8*/                   DFMA R12, -R28, R14, R16;                                   /* 0x2020000039c31e01 */
                                                                                               /* 0x2202b2c282804207 */
        /*15c8*/                   DFMA R10, R10, R12, R14;                                    /* 0x201c000030a29c01 */
        /*15d0*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0xc], PT;                 /* 0x160f4800311fdc80 */
        /*15d8*/                   SSY 0x1640;                                                 /* 0x6000000180001c07 */
        /*15e0*/                   FFMA R9, RZ, R29, R11;                                      /* 0x3016000077f25c00 */
        /*15e8*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*15f0*/               @P0 BRA CC.NEU, 0x1638;                                         /* 0x40000001000001a7 */
        /*15f8*/                   MOV R10, R16;                                               /* 0x2800000040029de4 */
                                                                                               /* 0x22b2b042b2b20047 */
        /*1608*/                   MOV R11, R17;                                               /* 0x280000004402dde4 */
        /*1610*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*1618*/                   MOV R9, R29;                                                /* 0x2800000074025de4 */
        /*1620*/                   CAL 0x17e0;                                                 /* 0x50000006e0011c07 */
        /*1628*/                   MOV R11, R10;                                               /* 0x280000002802dde4 */
        /*1630*/                   MOV R10, R9;                                                /* 0x2800000024029de4 */
        /*1638*/                   NOP.S;                                                      /* 0x4000000000001df4 */
                                                                                               /* 0x2292804200423047 */
        /*1648*/                   LD.E.64 R12, [R2+0x4e218];                                  /* 0x8400138860231ca5 */
        /*1650*/                   MUFU.RCP64H R15, R29;                                       /* 0xc800000019d3dc00 */
        /*1658*/                   LD.E.64 R26, [R2+0x5dc18];                                  /* 0x8400177060269ca5 */
        /*1660*/                   DMUL R24, R24, R30;                                         /* 0x5000000079861c01 */
        /*1668*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*1670*/                   SSY 0x1730;                                                 /* 0x60000002e0001c07 */
        /*1678*/                   DFMA R16, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c41e01 */
                                                                                               /* 0x2282820292929297 */
        /*1688*/                   DFMA R16, R16, R16, R16;                                    /* 0x2020000041041c01 */
        /*1690*/                   DFMA R18, R14, R16, R14;                                    /* 0x201c000040e49c01 */
        /*1698*/                   DMUL R14, R18, R24;                                         /* 0x5000000061239c01 */
        /*16a0*/                   DFMA R16, -R28, R14, R24;                                   /* 0x2030000039c41e01 */
        /*16a8*/                   DFMA R16, R18, R16, R14;                                    /* 0x201c000041241c01 */
        /*16b0*/                   FSET.GTU.AND RZ.CC, |R25|, c[0x2][0xc], PT;                 /* 0x160f4800319fdc80 */
        /*16b8*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
                                                                                               /* 0x22020042b04293f7 */
        /*16c8*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*16d0*/                   DADD R12, R10, R12;                                         /* 0x4800000030a31c01 */
        /*16d8*/                   ST.E.64 [R2+0x4e218], R12;                                  /* 0x9400138860231ca5 */
        /*16e0*/               @P0 BRA CC.NEU, 0x1728;                                         /* 0x40000001000001a7 */
        /*16e8*/                   MOV R10, R24;                                               /* 0x2800000060029de4 */
        /*16f0*/                   MOV R11, R25;                                               /* 0x280000006402dde4 */
        /*16f8*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
                                                                                               /* 0x22c2b2b2b042b2b7 */
        /*1708*/                   MOV R9, R29;                                                /* 0x2800000074025de4 */
        /*1710*/                   CAL 0x17e0;                                                 /* 0x5000000320011c07 */
        /*1718*/                   MOV R16, R9;                                                /* 0x2800000024041de4 */
        /*1720*/                   MOV R17, R10;                                               /* 0x2800000028045de4 */
        /*1728*/                   DADD.S R16, R26, R16;                                       /* 0x4800000041a41c11 */
        /*1730*/                   ST.E.64.S [R2+0x5dc18], R16;                                /* 0x9400177060241cb5 */
        /*1738*/                   IADD32I R44.CC, R44, -0x5;                                  /* 0x0fffffffeecb1c02 */
                                                                                               /* 0x2202c042c202c047 */
        /*1748*/                   IADD32I.X R43, R43, -0x1;                                   /* 0x0bfffffffebadc42 */
        /*1750*/                   IADD RZ.CC, -RZ, R44;                                       /* 0x48010000b3ffde03 */
        /*1758*/                   ISETP.NE.X.AND P0, PT, R43, RZ, PT;                         /* 0x1a8e0000feb1dc63 */
        /*1760*/                   IADD32I R4.CC, R4, 0x28;                                    /* 0x0c000000a0411c02 */
        /*1768*/                   IADD.X R5, RZ, R5;                                          /* 0x4800000017f15c43 */
        /*1770*/                   IADD32I R8.CC, R8, 0x5;                                     /* 0x0c00000014821c02 */
        /*1778*/                   IADD.X R37, RZ, R37;                                        /* 0x4800000097f95c43 */
                                                                                               /* 0x22c202c042c202c7 */
        /*1788*/                   IADD32I R7.CC, R7, 0x5;                                     /* 0x0c0000001471dc02 */
        /*1790*/                   IADD.X R38, RZ, R38;                                        /* 0x480000009bf99c43 */
        /*1798*/                   IADD32I R6.CC, R6, 0x5;                                     /* 0x0c00000014619c02 */
        /*17a0*/                   IADD.X R39, RZ, R39;                                        /* 0x480000009ff9dc43 */
        /*17a8*/                   IADD32I R0.CC, R0, 0x5;                                     /* 0x0c00000014001c02 */
        /*17b0*/                   IADD.X R40, RZ, R40;                                        /* 0x48000000a3fa1c43 */
        /*17b8*/                   IADD32I R32.CC, R32, 0x5;                                   /* 0x0c00000016081c02 */
                                                                                               /* 0x2200428042e2e047 */
        /*17c8*/                   IADD.X R41, RZ, R41;                                        /* 0x48000000a7fa5c43 */
        /*17d0*/               @P0 BRA 0x140;                                                  /* 0x4003ffa5a00001e7 */
        /*17d8*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*17e0*/                   LOP32I.AND R13, R9, 0x40000000;                             /* 0x3900000000935c02 */
        /*17e8*/                   MOV32I R45, 0x5ff00000;                                     /* 0x197fc000000b5de2 */
        /*17f0*/                   ISETP.LT.U32.AND P0, PT, R13, c[0x2][0x24], PT;             /* 0x188e480090d1dc03 */
        /*17f8*/                   PBK 0x1b40;                                                 /* 0x6800000d00001c07 */
                                                                                               /* 0x22928200428042b7 */
        /*1808*/                   MOV R13, R9;                                                /* 0x2800000024035de4 */
        /*1810*/                   SEL R14, R45, c[0x2][0x28], P0;                             /* 0x20004800a2d39c04 */
        /*1818*/                   LOP32I.AND R9, R11, 0x7f800000;                             /* 0x39fe000000b25c02 */
        /*1820*/                   MOV R15, R14;                                               /* 0x280000003803dde4 */
        /*1828*/                   ISETP.LT.U32.AND P0, PT, R9, c[0x2][0x2c], PT;              /* 0x188e4800b091dc03 */
        /*1830*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
        /*1838*/                   DMUL R16, R12, R14;                                         /* 0x5000000038c41c01 */
                                                                                               /* 0x2292804200428047 */
        /*1848*/                   MUFU.RCP64H R20, R17;                                       /* 0xc800000019151c00 */
        /*1850*/                   SEL R34, R45, c[0x2][0x28], P0;                             /* 0x20004800a2d89c04 */
        /*1858*/                   MOV R21, R20;                                               /* 0x2800000050055de4 */
        /*1860*/                   MOV R35, R34;                                               /* 0x280000008808dde4 */
        /*1868*/                   MOV32I R20, 0x1;                                            /* 0x1800000004051de2 */
        /*1870*/                   MOV R34, RZ;                                                /* 0x28000000fc089de4 */
        /*1878*/                   DFMA R18, -R16, R20, c[0x2][0x0];                           /* 0x2028880001049e01 */
                                                                                               /* 0x22d2929292925237 */
        /*1888*/                   DFMA R22, R18, R18, R18;                                    /* 0x2024000049259c01 */
        /*1890*/                   DMUL R18, R10, R34;                                         /* 0x5000000088a49c01 */
        /*1898*/                   DFMA R20, R20, R22, R20;                                    /* 0x2028000059451c01 */
        /*18a0*/                   DMUL R22, R18, R20;                                         /* 0x5000000051259c01 */
        /*18a8*/                   DFMA R16, -R16, R22, R18;                                   /* 0x2024000059041e01 */
        /*18b0*/                   DFMA R20, R20, R16, R22;                                    /* 0x202c000041451c01 */
        /*18b8*/                   DSETP.LEU.AND P0, PT, |R20|, RZ, PT;                        /* 0x1d8e0000fd41dc81 */
                                                                                               /* 0x2232804282c042b7 */
        /*18c8*/               @P0 BRA 0x1ab8;                                                 /* 0x40000007a00001e7 */
        /*18d0*/                   ISETP.GT.U32.AND P0, PT, R9, c[0x2][0x30], PT;              /* 0x1a0e4800c091dc03 */
        /*18d8*/                   DMUL R18, R14, R20;                                         /* 0x5000000050e49c01 */
        /*18e0*/                   SEL R16, R45, c[0x2][0x28], P0;                             /* 0x20004800a2d41c04 */
        /*18e8*/                   MOV R17, R16;                                               /* 0x2800000040045de4 */
        /*18f0*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*18f8*/                   DMUL R20, R20, R16;                                         /* 0x5000000041451c01 */
                                                                                               /* 0x227202d292523257 */
        /*1908*/                   DMUL R18, R16, R18;                                         /* 0x5000000049049c01 */
        /*1910*/                   DMUL R20, R14, R20;                                         /* 0x5000000050e51c01 */
        /*1918*/                   DFMA R14, R12, R18, -R10;                                   /* 0x2014000048c39d01 */
        /*1920*/                   DFMA R16, R12, R20, -R10;                                   /* 0x2014000050c41d01 */
        /*1928*/                   DSETP.GT.AND P0, PT, |R14|, |R16|, PT;                      /* 0x1a0e000040e1dcc1 */
        /*1930*/                   SEL R9, R21, R19, P0;                                       /* 0x200000004d525c04 */
        /*1938*/                   SEL R16, R20, R18, P0;                                      /* 0x2000000049441c04 */
                                                                                               /* 0x2200420042b042c7 */
        /*1948*/                   FSETP.GTU.AND P0, PT, |R9|, 1.469367938527859385e-39, PT;   /* 0x260ec0040091dc80 */
        /*1950*/                   MOV R17, R9;                                                /* 0x2800000024045de4 */
        /*1958*/               @P0 BRK;                                                        /* 0xa8000000000001e7 */
        /*1960*/                   FSETP.LT.AND P0, PT, |R11|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000b1dc80 */
        /*1968*/                   LOP32I.AND R19, R16, 0xfffffffe;                            /* 0x3bfffffff904dc02 */
        /*1970*/                   MOV32I R14, 0x58500000;                                     /* 0x1961400000039de2 */
        /*1978*/                   MOV R17, R9;                                                /* 0x2800000024045de4 */
                                                                                               /* 0x22023280428042a7 */
        /*1988*/                   LOP32I.OR R18, R16, 0x1;                                    /* 0x3800000005049c42 */
        /*1990*/                   SEL R14, R14, c[0x2][0x4], P0;                              /* 0x2000480010e39c04 */
        /*1998*/                   MOV R16, R18;                                               /* 0x2800000048041de4 */
        /*19a0*/                   MOV R15, R14;                                               /* 0x280000003803dde4 */
        /*19a8*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
        /*19b0*/                   DMUL R12, R12, R14;                                         /* 0x5000000038c31c01 */
        /*19b8*/                   DMUL R10, R10, R14;                                         /* 0x5000000038a29c01 */
                                                                                               /* 0x220282d292327207 */
        /*19c8*/                   MOV R14, R19;                                               /* 0x280000004c039de4 */
        /*19d0*/                   MOV R15, R9;                                                /* 0x280000002403dde4 */
        /*19d8*/                   DFMA R16, R16, R12, -R10;                                   /* 0x2014000031041d01 */
        /*19e0*/                   DFMA R14, R14, R12, -R10;                                   /* 0x2014000030e39d01 */
        /*19e8*/                   DSETP.GT.AND P0, PT, |R14|, |R16|, PT;                      /* 0x1a0e000040e1dcc1 */
        /*19f0*/                   SEL R14, R18, R19, P0;                                      /* 0x200000004d239c04 */
        /*19f8*/                   IADD32I R19.CC, R14, 0x1;                                   /* 0x0c00000004e4dc02 */
                                                                                               /* 0x22020042c20042b7 */
        /*1a08*/                   LOP32I.AND R15, R14, 0x1;                                   /* 0x3800000004e3dc02 */
        /*1a10*/                   IADD.X R18, RZ, R9;                                         /* 0x4800000027f49c43 */
        /*1a18*/                   IADD32I R16.CC, R14, -0x1;                                  /* 0x0ffffffffce41c02 */
        /*1a20*/                   ISETP.EQ.U32.AND P0, PT, R15, 0x1, PT;                      /* 0x190ec00004f1dc03 */
        /*1a28*/                   IADD32I.X R15, R9, -0x1;                                    /* 0x0bfffffffc93dc42 */
        /*1a30*/                   SEL R18, R9, R18, !P0;                                      /* 0x2010000048949c04 */
        /*1a38*/                   SEL R19, R14, R19, !P0;                                     /* 0x201000004ce4dc04 */
                                                                                               /* 0x2292621262020267 */
        /*1a48*/                   SEL R14, R16, R14, !P0;                                     /* 0x2010000039039c04 */
        /*1a50*/                   SEL R9, R15, R9, !P0;                                       /* 0x2010000024f25c04 */
        /*1a58*/                   MOV R16, R19;                                               /* 0x280000004c041de4 */
        /*1a60*/                   MOV R17, R18;                                               /* 0x2800000048045de4 */
        /*1a68*/                   MOV R15, R9;                                                /* 0x280000002403dde4 */
        /*1a70*/                   DFMA R16, R12, R16, -R10;                                   /* 0x2014000040c41d01 */
        /*1a78*/                   DFMA R10, R12, R14, -R10;                                   /* 0x2014000038c29d01 */
                                                                                               /* 0x22d2e042027202d7 */
        /*1a88*/                   DSETP.GT.AND P0, PT, |R16|, |R10|, PT;                      /* 0x1a0e00002901dcc1 */
        /*1a90*/                   SEL R14, R14, R19, P0;                                      /* 0x200000004ce39c04 */
        /*1a98*/                   SEL R9, R9, R18, P0;                                        /* 0x2000000048925c04 */
        /*1aa0*/                   MOV R16, R14;                                               /* 0x2800000038041de4 */
        /*1aa8*/                   MOV R17, R9;                                                /* 0x2800000024045de4 */
        /*1ab0*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1ab8*/                   DSETP.EQ.AND P0, PT, R20, RZ, PT;                           /* 0x190e0000fd41dc01 */
                                                                                               /* 0x2202d2d2804282b7 */
        /*1ac8*/               @P0 BRA 0x1b30;                                                 /* 0x40000001800001e7 */
        /*1ad0*/                   MUFU.RCP64H R14, R13;                                       /* 0xc800000018d39c00 */
        /*1ad8*/                   MOV R15, R14;                                               /* 0x280000003803dde4 */
        /*1ae0*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
        /*1ae8*/                   DSETP.GT.AND P1, PT, |R14|, RZ, PT;                         /* 0x1a0e0000fce3dc81 */
        /*1af0*/              @!P1 DSETP.NEU.AND P0, PT, |R12|, +INF , PT;                     /* 0x1e8edffc00c1e481 */
        /*1af8*/              @!P1 SEL R9, R12, R14, P0;                                       /* 0x2000000038c26404 */
                                                                                               /* 0x22b202e202820277 */
        /*1b08*/              @!P1 SEL R12, R13, R15, P0;                                      /* 0x200000003cd32404 */
        /*1b10*/              @!P1 MOV R14, R9;                                                /* 0x280000002403a5e4 */
        /*1b18*/              @!P1 MOV R15, R12;                                               /* 0x280000003003e5e4 */
        /*1b20*/                   DMUL R16, R10, R14;                                         /* 0x5000000038a41c01 */
        /*1b28*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1b30*/                   DMUL R16, R10, R12;                                         /* 0x5000000030a41c01 */
        /*1b38*/                   BRK;                                                        /* 0xa800000000001de7 */
                                                                                               /* 0x22b2b20042e20047 */
        /*1b48*/                   MOV R9, R16;                                                /* 0x2800000040025de4 */
        /*1b50*/                   MOV R10, R17;                                               /* 0x2800000044029de4 */
        /*1b58*/                   RET;                                                        /* 0x9000000000001de7 */
        /*1b60*/                   ISETP.LT.U32.AND P0, PT, R15, c[0x2][0x34], PT;             /* 0x188e4800d0f1dc03 */
        /*1b68*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*1b70*/                   PBK 0x1cb8;                                                 /* 0x6800000500001c07 */
        /*1b78*/               @P0 BRA 0x1bd0;                                                 /* 0x40000001400001e7 */
                                                                                               /* 0x220202d29282c297 */
        /*1b88*/                   DFMA.RM R12, R12, R14, R16;                                 /* 0x20a0000038c31c01 */
        /*1b90*/                   IADD32I R14.CC, R12, 0x1;                                   /* 0x0c00000004c39c02 */
        /*1b98*/                   IADD.X R15, RZ, R13;                                        /* 0x4800000037f3dc43 */
        /*1ba0*/                   DFMA.RP R10, -R12, R14, R10;                                /* 0x2114000038c29e01 */
        /*1ba8*/                   DSETP.GT.AND P0, PT, R10, RZ, PT;                           /* 0x1a0e0000fca1dc01 */
        /*1bb0*/                   SEL R12, R14, R12, P0;                                      /* 0x2000000030e31c04 */
        /*1bb8*/                   SEL R13, R15, R13, P0;                                      /* 0x2000000034f35c04 */
                                                                                               /* 0x22b20042c2b2d2e7 */
        /*1bc8*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1bd0*/                   DSETP.EQU.AND P0, PT, R10, RZ, PT;                          /* 0x1d0e0000fca1dc01 */
        /*1bd8*/               @P0 BRA 0x1ca8;                                                 /* 0x40000003200001e7 */
        /*1be0*/                   ISETP.LT.AND P0, PT, R11, RZ, PT;                           /* 0x188e0000fcb1dc23 */
        /*1be8*/               @P0 MOV32I R12, 0x0;                                            /* 0x18000000000301e2 */
        /*1bf0*/               @P0 MOV32I R13, 0xfff80000;                                     /* 0x1bffe000000341e2 */
        /*1bf8*/               @P0 BRK;                                                        /* 0xa8000000000001e7 */
                                                                                               /* 0x228042804202b2c7 */
        /*1c08*/                   ISETP.GT.AND P0, PT, R11, c[0x2][0x38], PT;                 /* 0x1a0e4800e0b1dc23 */
        /*1c10*/               @P0 BRA 0x1ca8;                                                 /* 0x40000002400001e7 */
        /*1c18*/                   DMUL R10, R10, 8.11296384146066816958e+31;                  /* 0x5000d1a400a29c01 */
        /*1c20*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
        /*1c28*/                   MOV32I R12, 0x0;                                            /* 0x1800000000031de2 */
        /*1c30*/                   MUFU.RSQ64H R15, R11;                                       /* 0xc80000001cb3dc00 */
        /*1c38*/                   MOV32I R13, 0x3fe00000;                                     /* 0x18ff800000035de2 */
                                                                                               /* 0x2282029292329297 */
        /*1c48*/                   DMUL R16, R14, R14;                                         /* 0x5000000038e41c01 */
        /*1c50*/                   DFMA R16, -R10, R16, c[0x2][0x0];                           /* 0x2020880000a41e01 */
        /*1c58*/                   DFMA R12, R16, 0.375, R12;                                  /* 0x2018cff601031c01 */
        /*1c60*/                   DMUL R16, R14, R16;                                         /* 0x5000000040e41c01 */
        /*1c68*/                   DFMA R14, R12, R16, R14;                                    /* 0x201c000040c39c01 */
        /*1c70*/                   DMUL R12, R10, R14;                                         /* 0x5000000038a31c01 */
        /*1c78*/                   IADD32I R15, R15, -0x100000;                                /* 0x0bffc00000f3dc02 */
                                                                                               /* 0x2202b202e0429297 */
        /*1c88*/                   DFMA R10, -R12, R12, R10;                                   /* 0x2014000030c29e01 */
        /*1c90*/                   DFMA R12, R14, R10, R12;                                    /* 0x2018000028e31c01 */
        /*1c98*/                   IADD32I R13, R13, -0x3500000;                               /* 0x0bf2c00000d35c02 */
        /*1ca0*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1ca8*/                   DADD R12, R10, R10;                                         /* 0x4800000028a31c01 */
        /*1cb0*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1cb8*/                   MOV R28, R12;                                               /* 0x2800000030071de4 */
                                                                                               /* 0x200000000002e047 */
        /*1cc8*/                   MOV R29, R13;                                               /* 0x2800000034075de4 */
        /*1cd0*/                   RET;                                                        /* 0x9000000000001de7 */
        /*1cd8*/                   BRA 0x1cd8;                                                 /* 0x4003ffffe0001de7 */
        /*1ce0*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*1ce8*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*1cf0*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*1cf8*/                   NOP;                                                        /* 0x4000000000001de4 */
		..........................................................................................................................................................


		Function : _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_11reset_forceEvEEE4callEvEUlvE_JEEEvT_DpT0_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                           /* 0x2280428042304307 */
        /*0008*/                   MOV R1, c[0x0][0x44];                   /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R7, 0x0;                         /* 0x180000000001dde2 */
        /*0018*/                   S2R R4, SR_TID.X;                       /* 0x2c00000084011c04 */
        /*0020*/                   MOV32I R6, 0x0;                         /* 0x1800000000019de2 */
        /*0028*/                   S2R R0, SR_CTAID.X;                     /* 0x2c00000094001c04 */
        /*0030*/                   MOV R2, R7;                             /* 0x280000001c009de4 */
        /*0038*/                   MOV R3, R6;                             /* 0x280000001800dde4 */
                                                                           /* 0x2282f2c2c3f28047 */
        /*0048*/                   LD.E.64 R2, [R2];                       /* 0x8400000000209ca5 */
        /*0050*/                   IMAD R4, R0, c[0x0][0x28], R4;          /* 0x20084000a0011ca3 */
        /*0058*/                   SHR R5, R4, 0x1f;                       /* 0x5800c0007c415c23 */
        /*0060*/                   ISUB RZ.CC, R4, R2;                     /* 0x48010000084fdd03 */
        /*0068*/                   ISETP.GE.U32.X.AND P0, PT, R5, R3, PT;  /* 0x1b0e00000c51dc43 */
        /*0070*/               @P0 EXIT;                                   /* 0x80000000000001e7 */
        /*0078*/                   MOV32I R0, 0x3;                         /* 0x180000000c001de2 */
                                                                           /* 0x2202b20242323047 */
        /*0088*/                   IADD32I RZ.CC, R0, -0x20;               /* 0x0fffffff800fdc02 */
        /*0090*/                   SHL R5, R5, 0x3;                        /* 0x6000c0000c515c03 */
        /*0098*/                   SHL R0, R4, -0x1d;                      /* 0x6000ffff8c401c03 */
        /*00a0*/                   SHR.U32 R2, R4, 0x1d;                   /* 0x5800c00074409c03 */
        /*00a8*/                   CSETP.SFT.AND P0, PT, CC, PT;           /* 0x040e00000151dc04 */
        /*00b0*/                   IMAD.U32.U32 R4.CC, R4, 0x8, R7;        /* 0x200fc00020411c03 */
        /*00b8*/               @P0 IADD R0, R5, R2;                        /* 0x4800000008500003 */
                                                                           /* 0x200002f042e28277 */
        /*00c8*/                   MOV R2, R4;                             /* 0x2800000010009de4 */
        /*00d0*/                   IADD.X R3, R0, R6;                      /* 0x480000001800dc43 */
        /*00d8*/                   ST.E.64 [R2+0x4e218], RZ;               /* 0x94001388602fdca5 */
        /*00e0*/                   ST.E.64 [R2+0x5dc18], RZ;               /* 0x94001770602fdca5 */
        /*00e8*/                   EXIT;                                   /* 0x8000000000001de7 */
        /*00f0*/                   BRA 0xf0;                               /* 0x4003ffffe0001de7 */
        /*00f8*/                   NOP;                                    /* 0x4000000000001de4 */
		...............................................................................................................................................


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
                                                                                              /* 0x229283f23042b047 */
        /*0088*/                   IADD.X R3, R0, R3;                                         /* 0x480000000c00dc43 */
        /*0090*/                   MOV32I R0, 0xfffffc18;                                     /* 0x1bfffff060001de2 */
        /*0098*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*00a0*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*00a8*/                   LD.E.64 R4, [R2+0x4e210];                                  /* 0x8400138840211ca5 */
        /*00b0*/                   MUFU.RCP64H R11, R7;                                       /* 0xc80000001872dc00 */
        /*00b8*/                   DFMA R8, -R6, R10, c[0x2][0x0];                            /* 0x2014880000621e01 */
                                                                                              /* 0x2282029292925237 */
        /*00c8*/                   DFMA R12, R8, R8, R8;                                      /* 0x2010000020831c01 */
        /*00d0*/                   DMUL R8, R4, c[0x0][0x148];                                /* 0x5000400520421c01 */
        /*00d8*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*00e0*/                   DMUL R4, R8, R10;                                          /* 0x5000000028811c01 */
        /*00e8*/                   DFMA R14, -R6, R4, R8;                                     /* 0x2010000010639e01 */
        /*00f0*/                   DFMA R10, R10, R14, R4;                                    /* 0x2008000038a29c01 */
        /*00f8*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
                                                                                              /* 0x2232b2b042b2c287 */
        /*0108*/                   FFMA R4, RZ, R7, R11;                                      /* 0x301600001ff11c00 */
        /*0110*/                   FSETP.GT.AND P0, PT, |R4|, c[0x2][0xc], PT;                /* 0x220e48003041dc80 */
        /*0118*/               @P0 BRA CC.NEU, 0x138;                                         /* 0x40000000600001a7 */
        /*0120*/                   MOV R4, R6;                                                /* 0x2800000018011de4 */
        /*0128*/                   MOV R5, R7;                                                /* 0x280000001c015de4 */
        /*0130*/                   CAL 0x978;                                                 /* 0x5000002100011c07 */
        /*0138*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
                                                                                              /* 0x22929283f2323047 */
        /*0148*/                   LD.E.64 R6, [R2+0x5dc10];                                  /* 0x8400177040219ca5 */
        /*0150*/                   MOV32I R8, 0x1;                                            /* 0x1800000004021de2 */
        /*0158*/                   LD.E.64 R24, [R2+0x2ee10];                                 /* 0x84000bb840261ca5 */
        /*0160*/                   LD.E.64 R22, [R2+0x3e810];                                 /* 0x84000fa040259ca5 */
        /*0168*/                   MUFU.RCP64H R9, R5;                                        /* 0xc800000018525c00 */
        /*0170*/                   DFMA R12, -R4, R8, c[0x2][0x0];                            /* 0x2010880000431e01 */
        /*0178*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
                                                                                              /* 0x2232523292929237 */
        /*0188*/                   DFMA R12, R8, R12, R8;                                     /* 0x2010000030831c01 */
        /*0190*/                   DMUL R8, R6, c[0x0][0x148];                                /* 0x5000400520621c01 */
        /*0198*/                   DMUL R6, R8, R12;                                          /* 0x5000000030819c01 */
        /*01a0*/                   DFMA R16, -R4, R6, R8;                                     /* 0x2010000018441e01 */
        /*01a8*/                   DFMA R16, R12, R16, R6;                                    /* 0x200c000040c41c01 */
        /*01b0*/                   DADD R24, R10, R24;                                        /* 0x4800000060a61c01 */
        /*01b8*/                   FFMA R6, RZ, R5, R17;                                      /* 0x3022000017f19c00 */
                                                                                              /* 0x22b042b2b2c24047 */
        /*01c8*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*01d0*/                   ST.E.64 [R2+0x2ee10], R24;                                 /* 0x94000bb840261ca5 */
        /*01d8*/                   FSETP.GT.AND P0, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003061dc80 */
        /*01e0*/               @P0 BRA CC.NEU, 0x200;                                         /* 0x40000000600001a7 */
        /*01e8*/                   CAL 0x978;                                                 /* 0x5000001e20011c07 */
        /*01f0*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
        /*01f8*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
                                                                                              /* 0x22e2e23230423047 */
        /*0208*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*0210*/                   DADD R16, R22, R16;                                        /* 0x4800000041641c01 */
        /*0218*/                   LD.E.64 R12, [R2+0x4e210];                                 /* 0x8400138840231ca5 */
        /*0220*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0228*/                   LD.E.64 R6, [R2+0xfa10];                                   /* 0x840003e840219ca5 */
        /*0230*/                   LD.E.64 R8, [R2+0x1f410];                                  /* 0x840007d040221ca5 */
        /*0238*/                   ST.E.64 [R2+0x3e810], R16;                                 /* 0x94000fa040241ca5 */
                                                                                              /* 0x22929232929283f7 */
        /*0248*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
        /*0250*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*0258*/                   DFMA R18, -R4, R14, c[0x2][0x0];                           /* 0x201c880000449e01 */
        /*0260*/                   DFMA R18, R18, R18, R18;                                   /* 0x2024000049249c01 */
        /*0268*/                   DFMA R18, R14, R18, R14;                                   /* 0x201c000048e49c01 */
        /*0270*/                   DMUL R14, R12, c[0x0][0x148];                              /* 0x5000400520c39c01 */
        /*0278*/                   DMUL R32, R14, R18;                                        /* 0x5000000048e81c01 */
                                                                                              /* 0x22d2004252323297 */
        /*0288*/                   DFMA R12, -R4, R32, R14;                                   /* 0x201c000080431e01 */
        /*0290*/                   DFMA R18, R18, R12, R32;                                   /* 0x2040000031249c01 */
        /*0298*/                   DFMA R6, R24, c[0x0][0x148], R6;                           /* 0x200c400521819c01 */
        /*02a0*/                   DFMA R8, R10, c[0x0][0x148], R8;                           /* 0x2010400520a21c01 */
        /*02a8*/                   FFMA R10, RZ, R5, R19;                                     /* 0x3026000017f29c00 */
        /*02b0*/                   ST.E.64 [R2+0xfa10], R6;                                   /* 0x940003e840219ca5 */
        /*02b8*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
                                                                                              /* 0x2202b2b042b2c047 */
        /*02c8*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*02d0*/                   ST.E.64 [R2+0x1f410], R8;                                  /* 0x940007d040221ca5 */
        /*02d8*/               @P0 BRA CC.NEU, 0x310;                                         /* 0x40000000c00001a7 */
        /*02e0*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*02e8*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
        /*02f0*/                   CAL 0x978;                                                 /* 0x5000001a00011c07 */
        /*02f8*/                   MOV R18, R10;                                              /* 0x2800000028049de4 */
                                                                                              /* 0x2283f232323042b7 */
        /*0308*/                   MOV R19, R11;                                              /* 0x280000002c04dde4 */
        /*0310*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*0318*/                   MOV32I R8, 0x1;                                            /* 0x1800000004021de2 */
        /*0320*/                   LD.E.64 R6, [R2+0x5dc10];                                  /* 0x8400177040219ca5 */
        /*0328*/                   LD.E.64 R24, [R2+0x2ee10];                                 /* 0x84000bb840261ca5 */
        /*0330*/                   LD.E.64 R22, [R2+0x3e810];                                 /* 0x84000fa040259ca5 */
        /*0338*/                   MUFU.RCP64H R9, R5;                                        /* 0xc800000018525c00 */
                                                                                              /* 0x2232929292329297 */
        /*0348*/                   DFMA R10, -R4, R8, c[0x2][0x0];                            /* 0x2010880000429e01 */
        /*0350*/                   DFMA R10, R10, R10, R10;                                   /* 0x2014000028a29c01 */
        /*0358*/                   DFMA R10, R8, R10, R8;                                     /* 0x2010000028829c01 */
        /*0360*/                   DMUL R8, R6, c[0x0][0x148];                                /* 0x5000400520621c01 */
        /*0368*/                   DMUL R6, R8, R10;                                          /* 0x5000000028819c01 */
        /*0370*/                   DFMA R12, -R4, R6, R8;                                     /* 0x2010000018431e01 */
        /*0378*/                   DFMA R10, R10, R12, R6;                                    /* 0x200c000030a29c01 */
                                                                                              /* 0x22b2b2c272004297 */
        /*0388*/                   DADD R24, R18, R24;                                        /* 0x4800000061261c01 */
        /*0390*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*0398*/                   ST.E.64 [R2+0x2ee10], R24;                                 /* 0x94000bb840261ca5 */
        /*03a0*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*03a8*/                   FSETP.GT.AND P0, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003061dc80 */
        /*03b0*/               @P0 BRA CC.NEU, 0x3c0;                                         /* 0x40000000200001a7 */
        /*03b8*/                   CAL 0x978;                                                 /* 0x50000016e0011c07 */
                                                                                              /* 0x22e2e23230423047 */
        /*03c8*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*03d0*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*03d8*/                   LD.E.64 R12, [R2+0x4e210];                                 /* 0x8400138840231ca5 */
        /*03e0*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*03e8*/                   LD.E.64 R6, [R2+0xfa10];                                   /* 0x840003e840219ca5 */
        /*03f0*/                   LD.E.64 R8, [R2+0x1f410];                                  /* 0x840007d040221ca5 */
        /*03f8*/                   ST.E.64 [R2+0x3e810], R22;                                 /* 0x94000fa040259ca5 */
                                                                                              /* 0x22929232929283f7 */
        /*0408*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
        /*0410*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*0418*/                   DFMA R16, -R4, R14, c[0x2][0x0];                           /* 0x201c880000441e01 */
        /*0420*/                   DFMA R16, R16, R16, R16;                                   /* 0x2020000041041c01 */
        /*0428*/                   DFMA R16, R14, R16, R14;                                   /* 0x201c000040e41c01 */
        /*0430*/                   DMUL R14, R12, c[0x0][0x148];                              /* 0x5000400520c39c01 */
        /*0438*/                   DMUL R18, R14, R16;                                        /* 0x5000000040e49c01 */
                                                                                              /* 0x22d2004252323297 */
        /*0448*/                   DFMA R12, -R4, R18, R14;                                   /* 0x201c000048431e01 */
        /*0450*/                   DFMA R16, R16, R12, R18;                                   /* 0x2024000031041c01 */
        /*0458*/                   DFMA R6, R24, c[0x0][0x148], R6;                           /* 0x200c400521819c01 */
        /*0460*/                   DFMA R8, R10, c[0x0][0x148], R8;                           /* 0x2010400520a21c01 */
        /*0468*/                   FFMA R10, RZ, R5, R17;                                     /* 0x3022000017f29c00 */
        /*0470*/                   ST.E.64 [R2+0xfa10], R6;                                   /* 0x940003e840219ca5 */
        /*0478*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
                                                                                              /* 0x2202b2b042b2c047 */
        /*0488*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*0490*/                   ST.E.64 [R2+0x1f410], R8;                                  /* 0x940007d040221ca5 */
        /*0498*/               @P0 BRA CC.NEU, 0x4d0;                                         /* 0x40000000c00001a7 */
        /*04a0*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*04a8*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
        /*04b0*/                   CAL 0x978;                                                 /* 0x5000001300011c07 */
        /*04b8*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
                                                                                              /* 0x2283f232323042b7 */
        /*04c8*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*04d0*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*04d8*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*04e0*/                   LD.E.64 R8, [R2+0x5dc10];                                  /* 0x8400177040221ca5 */
        /*04e8*/                   LD.E.64 R6, [R2+0x2ee10];                                  /* 0x84000bb840219ca5 */
        /*04f0*/                   LD.E.64 R22, [R2+0x3e810];                                 /* 0x84000fa040259ca5 */
        /*04f8*/                   MUFU.RCP64H R11, R5;                                       /* 0xc80000001852dc00 */
                                                                                              /* 0x2232929292523297 */
        /*0508*/                   DFMA R12, -R4, R10, c[0x2][0x0];                           /* 0x2014880000431e01 */
        /*0510*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*0518*/                   DMUL R8, R8, c[0x0][0x148];                                /* 0x5000400520821c01 */
        /*0520*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*0528*/                   DMUL R14, R8, R10;                                         /* 0x5000000028839c01 */
        /*0530*/                   DFMA R12, -R4, R14, R8;                                    /* 0x2010000038431e01 */
        /*0538*/                   DFMA R10, R10, R12, R14;                                   /* 0x201c000030a29c01 */
                                                                                              /* 0x22b2b2c272004297 */
        /*0548*/                   DADD R24, R16, R6;                                         /* 0x4800000019061c01 */
        /*0550*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*0558*/                   ST.E.64 [R2+0x2ee10], R24;                                 /* 0x94000bb840261ca5 */
        /*0560*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*0568*/                   FSETP.GT.AND P0, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003061dc80 */
        /*0570*/               @P0 BRA CC.NEU, 0x580;                                         /* 0x40000000200001a7 */
        /*0578*/                   CAL 0x978;                                                 /* 0x5000000fe0011c07 */
                                                                                              /* 0x22e2e23230423047 */
        /*0588*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*0590*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*0598*/                   LD.E.64 R12, [R2+0x4e210];                                 /* 0x8400138840231ca5 */
        /*05a0*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*05a8*/                   LD.E.64 R6, [R2+0xfa10];                                   /* 0x840003e840219ca5 */
        /*05b0*/                   LD.E.64 R8, [R2+0x1f410];                                  /* 0x840007d040221ca5 */
        /*05b8*/                   ST.E.64 [R2+0x3e810], R22;                                 /* 0x94000fa040259ca5 */
                                                                                              /* 0x22929232929283f7 */
        /*05c8*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
        /*05d0*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*05d8*/                   DFMA R16, -R4, R14, c[0x2][0x0];                           /* 0x201c880000441e01 */
        /*05e0*/                   DFMA R16, R16, R16, R16;                                   /* 0x2020000041041c01 */
        /*05e8*/                   DFMA R16, R14, R16, R14;                                   /* 0x201c000040e41c01 */
        /*05f0*/                   DMUL R14, R12, c[0x0][0x148];                              /* 0x5000400520c39c01 */
        /*05f8*/                   DMUL R18, R14, R16;                                        /* 0x5000000040e49c01 */
                                                                                              /* 0x22d2004252323297 */
        /*0608*/                   DFMA R12, -R4, R18, R14;                                   /* 0x201c000048431e01 */
        /*0610*/                   DFMA R16, R16, R12, R18;                                   /* 0x2024000031041c01 */
        /*0618*/                   DFMA R6, R24, c[0x0][0x148], R6;                           /* 0x200c400521819c01 */
        /*0620*/                   DFMA R8, R10, c[0x0][0x148], R8;                           /* 0x2010400520a21c01 */
        /*0628*/                   FFMA R10, RZ, R5, R17;                                     /* 0x3022000017f29c00 */
        /*0630*/                   ST.E.64 [R2+0xfa10], R6;                                   /* 0x940003e840219ca5 */
        /*0638*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
                                                                                              /* 0x2202b2b042b2c047 */
        /*0648*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*0650*/                   ST.E.64 [R2+0x1f410], R8;                                  /* 0x940007d040221ca5 */
        /*0658*/               @P0 BRA CC.NEU, 0x690;                                         /* 0x40000000c00001a7 */
        /*0660*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0668*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
        /*0670*/                   CAL 0x978;                                                 /* 0x5000000c00011c07 */
        /*0678*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
                                                                                              /* 0x2283f232323042b7 */
        /*0688*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*0690*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*0698*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*06a0*/                   LD.E.64 R8, [R2+0x5dc10];                                  /* 0x8400177040221ca5 */
        /*06a8*/                   LD.E.64 R6, [R2+0x2ee10];                                  /* 0x84000bb840219ca5 */
        /*06b0*/                   LD.E.64 R22, [R2+0x3e810];                                 /* 0x84000fa040259ca5 */
        /*06b8*/                   MUFU.RCP64H R11, R5;                                       /* 0xc80000001852dc00 */
                                                                                              /* 0x2232929292523297 */
        /*06c8*/                   DFMA R12, -R4, R10, c[0x2][0x0];                           /* 0x2014880000431e01 */
        /*06d0*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*06d8*/                   DMUL R8, R8, c[0x0][0x148];                                /* 0x5000400520821c01 */
        /*06e0*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*06e8*/                   DMUL R14, R8, R10;                                         /* 0x5000000028839c01 */
        /*06f0*/                   DFMA R12, -R4, R14, R8;                                    /* 0x2010000038431e01 */
        /*06f8*/                   DFMA R10, R10, R12, R14;                                   /* 0x201c000030a29c01 */
                                                                                              /* 0x22b2b2c272004297 */
        /*0708*/                   DADD R24, R16, R6;                                         /* 0x4800000019061c01 */
        /*0710*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*0718*/                   ST.E.64 [R2+0x2ee10], R24;                                 /* 0x94000bb840261ca5 */
        /*0720*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*0728*/                   FSETP.GT.AND P0, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003061dc80 */
        /*0730*/               @P0 BRA CC.NEU, 0x740;                                         /* 0x40000000200001a7 */
        /*0738*/                   CAL 0x978;                                                 /* 0x50000008e0011c07 */
                                                                                              /* 0x22e2e23230423047 */
        /*0748*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*0750*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*0758*/                   LD.E.64 R12, [R2+0x4e210];                                 /* 0x8400138840231ca5 */
        /*0760*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0768*/                   LD.E.64 R6, [R2+0xfa10];                                   /* 0x840003e840219ca5 */
        /*0770*/                   LD.E.64 R8, [R2+0x1f410];                                  /* 0x840007d040221ca5 */
        /*0778*/                   ST.E.64 [R2+0x3e810], R22;                                 /* 0x94000fa040259ca5 */
                                                                                              /* 0x22929232929283f7 */
        /*0788*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
        /*0790*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*0798*/                   DFMA R16, -R4, R14, c[0x2][0x0];                           /* 0x201c880000441e01 */
        /*07a0*/                   DFMA R16, R16, R16, R16;                                   /* 0x2020000041041c01 */
        /*07a8*/                   DFMA R16, R14, R16, R14;                                   /* 0x201c000040e41c01 */
        /*07b0*/                   DMUL R14, R12, c[0x0][0x148];                              /* 0x5000400520c39c01 */
        /*07b8*/                   DMUL R18, R14, R16;                                        /* 0x5000000040e49c01 */
                                                                                              /* 0x22d2004252323297 */
        /*07c8*/                   DFMA R12, -R4, R18, R14;                                   /* 0x201c000048431e01 */
        /*07d0*/                   DFMA R16, R16, R12, R18;                                   /* 0x2024000031041c01 */
        /*07d8*/                   DFMA R6, R24, c[0x0][0x148], R6;                           /* 0x200c400521819c01 */
        /*07e0*/                   DFMA R8, R10, c[0x0][0x148], R8;                           /* 0x2010400520a21c01 */
        /*07e8*/                   FFMA R10, RZ, R5, R17;                                     /* 0x3022000017f29c00 */
        /*07f0*/                   ST.E.64 [R2+0xfa10], R6;                                   /* 0x940003e840219ca5 */
        /*07f8*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
                                                                                              /* 0x2202b2b042b2c047 */
        /*0808*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*0810*/                   ST.E.64 [R2+0x1f410], R8;                                  /* 0x940007d040221ca5 */
        /*0818*/               @P0 BRA CC.NEU, 0x850;                                         /* 0x40000000c00001a7 */
        /*0820*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0828*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
        /*0830*/                   CAL 0x978;                                                 /* 0x5000000500011c07 */
        /*0838*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
                                                                                              /* 0x2283f232323042b7 */
        /*0848*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*0850*/                   LD.E.64 R4, [R2+0x10];                                     /* 0x8400000040211ca5 */
        /*0858*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*0860*/                   LD.E.64 R8, [R2+0x5dc10];                                  /* 0x8400177040221ca5 */
        /*0868*/                   LD.E.64 R6, [R2+0x2ee10];                                  /* 0x84000bb840219ca5 */
        /*0870*/                   LD.E.64 R22, [R2+0x3e810];                                 /* 0x84000fa040259ca5 */
        /*0878*/                   MUFU.RCP64H R11, R5;                                       /* 0xc80000001852dc00 */
                                                                                              /* 0x2232929292523297 */
        /*0888*/                   DFMA R12, -R4, R10, c[0x2][0x0];                           /* 0x2014880000431e01 */
        /*0890*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*0898*/                   DMUL R8, R8, c[0x0][0x148];                                /* 0x5000400520821c01 */
        /*08a0*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*08a8*/                   DMUL R14, R8, R10;                                         /* 0x5000000028839c01 */
        /*08b0*/                   DFMA R12, -R4, R14, R8;                                    /* 0x2010000038431e01 */
        /*08b8*/                   DFMA R10, R10, R12, R14;                                   /* 0x201c000030a29c01 */
                                                                                              /* 0x22b2b2c272004297 */
        /*08c8*/                   DADD R24, R16, R6;                                         /* 0x4800000019061c01 */
        /*08d0*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*08d8*/                   ST.E.64 [R2+0x2ee10], R24;                                 /* 0x94000bb840261ca5 */
        /*08e0*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*08e8*/                   FSETP.GT.AND P0, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003061dc80 */
        /*08f0*/               @P0 BRA CC.NEU, 0x900;                                         /* 0x40000000200001a7 */
        /*08f8*/                   CAL 0x978;                                                 /* 0x50000001e0011c07 */
                                                                                              /* 0x23f2e29042804207 */
        /*0908*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*0910*/                   LD.E.64 R4, [R2+0xfa10];                                   /* 0x840003e840211ca5 */
        /*0918*/                   IADD32I R0, R0, 0x5;                                       /* 0x0800000014001c02 */
        /*0920*/                   LD.E.64 R6, [R2+0x1f410];                                  /* 0x840007d040219ca5 */
        /*0928*/                   ISETP.NE.AND P0, PT, R0, RZ, PT;                           /* 0x1a8e0000fc01dc23 */
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
        /*0988*/                   MOV32I R12, 0x5ff00000;                                    /* 0x197fc00000031de2 */
        /*0990*/                   LOP32I.AND R13, R9, 0x7f800000;                            /* 0x39fe000000935c02 */
        /*0998*/                   ISETP.LT.U32.AND P0, PT, R6, c[0x2][0x10], PT;             /* 0x188e48004061dc03 */
        /*09a0*/                   MOV R6, RZ;                                                /* 0x28000000fc019de4 */
        /*09a8*/                   SEL R7, R12, c[0x2][0x14], P0;                             /* 0x2000480050c1dc04 */
        /*09b0*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*09b8*/                   DMUL R10, R4, R6;                                          /* 0x5000000018429c01 */
                                                                                              /* 0x225232928042c047 */
        /*09c8*/                   ISETP.LT.U32.AND P0, PT, R13, c[0x2][0x18], PT;            /* 0x188e480060d1dc03 */
        /*09d0*/                   MOV R18, RZ;                                               /* 0x28000000fc049de4 */
        /*09d8*/                   MUFU.RCP64H R15, R11;                                      /* 0xc800000018b3dc00 */
        /*09e0*/                   SEL R19, R12, c[0x2][0x14], P0;                            /* 0x2000480050c4dc04 */
        /*09e8*/                   DFMA R20, -R10, R14, c[0x2][0x0];                          /* 0x201c880000a51e01 */
        /*09f0*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*09f8*/                   DMUL R16, R8, R18;                                         /* 0x5000000048841c01 */
                                                                                              /* 0x2202b2d292929297 */
        /*0a08*/                   DFMA R14, R14, R20, R14;                                   /* 0x201c000050e39c01 */
        /*0a10*/                   DMUL R20, R16, R14;                                        /* 0x5000000039051c01 */
        /*0a18*/                   DFMA R10, -R10, R20, R16;                                  /* 0x2020000050a29e01 */
        /*0a20*/                   DFMA R10, R14, R10, R20;                                   /* 0x2028000028e29c01 */
        /*0a28*/                   DSETP.LEU.AND P0, PT, |R10|, RZ, PT;                       /* 0x1d8e0000fca1dc81 */
        /*0a30*/               @P0 BRA 0xbd8;                                                 /* 0x40000006800001e7 */
        /*0a38*/                   ISETP.GT.U32.AND P0, PT, R13, c[0x2][0x1c], PT;            /* 0x1a0e480070d1dc03 */
                                                                                              /* 0x22523252328202b7 */
        /*0a48*/                   DMUL R14, R6, R10;                                         /* 0x5000000028639c01 */
        /*0a50*/                   SEL R13, R12, c[0x2][0x14], P0;                            /* 0x2000480050c35c04 */
        /*0a58*/                   MOV R12, RZ;                                               /* 0x28000000fc031de4 */
        /*0a60*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0a68*/                   DMUL R10, R12, R14;                                        /* 0x5000000038c29c01 */
        /*0a70*/                   DMUL R6, R6, R16;                                          /* 0x5000000040619c01 */
        /*0a78*/                   DFMA R12, R4, R10, -R8;                                    /* 0x2010000028431d01 */
                                                                                              /* 0x2202b2c27202d297 */
        /*0a88*/                   DFMA R14, R4, R6, -R8;                                     /* 0x2010000018439d01 */
        /*0a90*/                   DSETP.GT.AND P0, PT, |R12|, |R14|, PT;                     /* 0x1a0e000038c1dcc1 */
        /*0a98*/                   SEL R7, R7, R11, P0;                                       /* 0x200000002c71dc04 */
        /*0aa0*/                   SEL R6, R6, R10, P0;                                       /* 0x2000000028619c04 */
        /*0aa8*/                   FSETP.GTU.AND P0, PT, |R7|, 1.469367938527859385e-39, PT;  /* 0x260ec0040071dc80 */
        /*0ab0*/               @P0 BRA 0xc50;                                                 /* 0x40000006600001e7 */
        /*0ab8*/                   FSETP.LT.AND P0, PT, |R9|, 1.5046327690525280102e-36, PT;  /* 0x208ec1000091dc80 */
                                                                                              /* 0x2232829200420047 */
        /*0ac8*/                   MOV32I R15, 0x58500000;                                    /* 0x196140000003dde2 */
        /*0ad0*/                   LOP32I.AND R10, R6, 0xfffffffe;                            /* 0x3bfffffff8629c02 */
        /*0ad8*/                   MOV R14, RZ;                                               /* 0x28000000fc039de4 */
        /*0ae0*/                   MOV R11, R7;                                               /* 0x280000001c02dde4 */
        /*0ae8*/                   LOP32I.OR R6, R6, 0x1;                                     /* 0x3800000004619c42 */
        /*0af0*/                   SEL R15, R15, c[0x2][0x4], P0;                             /* 0x2000480010f3dc04 */
        /*0af8*/                   DMUL R4, R4, R14;                                          /* 0x5000000038411c01 */
                                                                                              /* 0x22c04282d2923297 */
        /*0b08*/                   DMUL R8, R8, R14;                                          /* 0x5000000038821c01 */
        /*0b10*/                   DFMA R12, R10, R4, -R8;                                    /* 0x2010000010a31d01 */
        /*0b18*/                   DFMA R14, R6, R4, -R8;                                     /* 0x2010000010639d01 */
        /*0b20*/                   DSETP.GT.AND P0, PT, |R12|, |R14|, PT;                     /* 0x1a0e000038c1dcc1 */
        /*0b28*/                   SEL R11, R6, R10, P0;                                      /* 0x200000002862dc04 */
        /*0b30*/                   IADD32I R13.CC, R11, 0x1;                                  /* 0x0c00000004b35c02 */
        /*0b38*/                   LOP32I.AND R10, R11, 0x1;                                  /* 0x3800000004b29c02 */
                                                                                              /* 0x2272004202b20047 */
        /*0b48*/                   ISETP.EQ.U32.AND P0, PT, R10, 0x1, PT;                     /* 0x190ec00004a1dc03 */
        /*0b50*/                   IADD.X R12, RZ, R7;                                        /* 0x480000001ff31c43 */
        /*0b58*/                   IADD32I R6.CC, R11, -0x1;                                  /* 0x0ffffffffcb19c02 */
        /*0b60*/                   SEL R13, R11, R13, !P0;                                    /* 0x2010000034b35c04 */
        /*0b68*/                   IADD32I.X R10, R7, -0x1;                                   /* 0x0bfffffffc729c42 */
        /*0b70*/                   SEL R12, R7, R12, !P0;                                     /* 0x2010000030731c04 */
        /*0b78*/                   SEL R6, R6, R11, !P0;                                      /* 0x201000002c619c04 */
                                                                                              /* 0x2202d29232820207 */
        /*0b88*/                   SEL R7, R10, R7, !P0;                                      /* 0x201000001ca1dc04 */
        /*0b90*/                   MOV R10, R13;                                              /* 0x2800000034029de4 */
        /*0b98*/                   MOV R11, R12;                                              /* 0x280000003002dde4 */
        /*0ba0*/                   DFMA R10, R4, R10, -R8;                                    /* 0x2010000028429d01 */
        /*0ba8*/                   DFMA R4, R4, R6, -R8;                                      /* 0x2010000018411d01 */
        /*0bb0*/                   DSETP.GT.AND P0, PT, |R10|, |R4|, PT;                      /* 0x1a0e000010a1dcc1 */
        /*0bb8*/                   SEL R6, R6, R13, P0;                                       /* 0x2000000034619c04 */
                                                                                              /* 0x22d28042b2d2e047 */
        /*0bc8*/                   SEL R7, R7, R12, P0;                                       /* 0x200000003071dc04 */
        /*0bd0*/                   BRA 0xc50;                                                 /* 0x40000001e0001de7 */
        /*0bd8*/                   DSETP.EQ.AND P0, PT, R10, RZ, PT;                          /* 0x190e0000fca1dc01 */
        /*0be0*/               @P0 BRA 0xc40;                                                 /* 0x40000001600001e7 */
        /*0be8*/                   MUFU.RCP64H R7, R5;                                        /* 0xc80000001851dc00 */
        /*0bf0*/                   MOV R6, RZ;                                                /* 0x28000000fc019de4 */
        /*0bf8*/                   DSETP.GT.AND P1, PT, |R6|, RZ, PT;                         /* 0x1a0e0000fc63dc81 */
                                                                                              /* 0x22e20282027202d7 */
        /*0c08*/              @!P1 DSETP.NEU.AND P0, PT, |R4|, +INF , PT;                     /* 0x1e8edffc0041e481 */
        /*0c10*/              @!P1 SEL R4, R4, R6, P0;                                        /* 0x2000000018412404 */
        /*0c18*/              @!P1 SEL R5, R5, R7, P0;                                        /* 0x200000001c516404 */
        /*0c20*/              @!P1 MOV R6, R4;                                                /* 0x280000001001a5e4 */
        /*0c28*/              @!P1 MOV R7, R5;                                                /* 0x280000001401e5e4 */
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
        /*02d8*/                   MOV32I R0, 0x5ff00000;                                      /* 0x197fc00000001de2 */
        /*02e0*/                   MOV R12, RZ;                                                /* 0x28000000fc031de4 */
        /*02e8*/                   SEL R13, R0, c[0x2][0x14], P0;                              /* 0x2000480050035c04 */
        /*02f0*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
        /*02f8*/                   DMUL R16, R10, R12;                                         /* 0x5000000030a41c01 */
                                                                                               /* 0x225232928042c047 */
        /*0308*/                   ISETP.LT.U32.AND P0, PT, R26, c[0x2][0x18], PT;             /* 0x188e480061a1dc03 */
        /*0310*/                   MOV R24, RZ;                                                /* 0x28000000fc061de4 */
        /*0318*/                   MUFU.RCP64H R19, R17;                                       /* 0xc80000001914dc00 */
        /*0320*/                   SEL R25, R0, c[0x2][0x14], P0;                              /* 0x2000480050065c04 */
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
        /*0398*/                   SEL R17, R0, c[0x2][0x14], P0;                              /* 0x2000480050045c04 */
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
        /*0408*/                   MOV32I R17, 0x58500000;                                     /* 0x1961400000045de2 */
        /*0410*/                   LOP32I.AND R0, R12, 0xfffffffe;                             /* 0x3bfffffff8c01c02 */
        /*0418*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*0420*/                   LOP32I.OR R20, R12, 0x1;                                    /* 0x3800000004c51c42 */
        /*0428*/                   SEL R17, R17, c[0x2][0x4], P0;                              /* 0x2000480011145c04 */
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
        /*0010*/                   MOV R5, c[0x0][0x140];                                      /* 0x2800400500015de4 */
        /*0018*/                   MOV R0, c[0x0][0x144];                                      /* 0x2800400510001de4 */
        /*0020*/                   ISUB RZ.CC, R5, c[0x0][0x148];                              /* 0x48014005205fdd03 */
        /*0028*/                   ISETP.EQ.X.AND P0, PT, R0, c[0x0][0x14c], PT;               /* 0x190e40053001dc63 */
        /*0030*/               @P0 EXIT;                                                       /* 0x80000000000001e7 */
        /*0038*/                   MOV32I R2, 0x3;                                             /* 0x180000000c009de2 */
                                                                                               /* 0x2242304220420047 */
        /*0048*/                   IADD32I RZ.CC, R2, -0x20;                                   /* 0x0fffffff802fdc02 */
        /*0050*/                   SHL R23, R5, 0x3;                                           /* 0x6000c0000c55dc03 */
        /*0058*/                   MOV32I R22, 0x0;                                            /* 0x1800000000059de2 */
        /*0060*/                   MOV R6, c[0x0][0x148];                                      /* 0x2800400520019de4 */
        /*0068*/                   SHL R25, R5, -0x1d;                                         /* 0x6000ffff8c565c03 */
        /*0070*/                   MOV R4, c[0x0][0x14c];                                      /* 0x2800400530011de4 */
        /*0078*/                   SHL R0, R0, 0x3;                                            /* 0x6000c0000c001c03 */
                                                                                               /* 0x2272020042720237 */
        /*0088*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*0090*/                   SHR.U32 R5, R5, 0x1d;                                       /* 0x5800c00074515c03 */
        /*0098*/                   IADD R10.CC, R23, R22;                                      /* 0x4801000059729c03 */
        /*00a0*/                   SHL R2, R6, 0x3;                                            /* 0x6000c0000c609c03 */
        /*00a8*/               @P0 IADD R25, R0, R5;                                           /* 0x4800000014064003 */
        /*00b0*/                   MOV32I R0, 0x0;                                             /* 0x1800000000001de2 */
        /*00b8*/                   SHL R4, R4, 0x3;                                            /* 0x6000c0000c411c03 */
                                                                                               /* 0x22e0427200428047 */
        /*00c8*/                   SHL R3, R6, -0x1d;                                          /* 0x6000ffff8c60dc03 */
        /*00d0*/                   IADD.X R11, R25, R0;                                        /* 0x480000000192dc43 */
        /*00d8*/                   SHR.U32 R5, R6, 0x1d;                                       /* 0x5800c00074615c03 */
        /*00e0*/                   LD.E.64 R20, [R10+0xfa10];                                  /* 0x840003e840a51ca5 */
        /*00e8*/                   IADD R14.CC, R22, R2;                                       /* 0x4801000009639c03 */
        /*00f0*/               @P0 IADD R3, R4, R5;                                            /* 0x480000001440c003 */
        /*00f8*/                   LD.E.64 R4, [R10+0x1f410];                                  /* 0x840007d040a11ca5 */
                                                                                               /* 0x225233f230428047 */
        /*0108*/                   IADD.X R15, R0, R3;                                         /* 0x480000000c03dc43 */
        /*0110*/                   MOV32I R10, 0x0;                                            /* 0x1800000000029de2 */
        /*0118*/                   LD.E.64 R8, [R14+0x1f410];                                  /* 0x840007d040e21ca5 */
        /*0120*/                   MOV32I R11, 0x3fe00000;                                     /* 0x18ff80000002dde2 */
        /*0128*/                   LD.E.64 R6, [R14+0xfa10];                                   /* 0x840003e840e19ca5 */
        /*0130*/                   DADD R4, -R4, R8;                                           /* 0x4800000020411e01 */
        /*0138*/                   DADD R20, -R20, R6;                                         /* 0x4800000019451e01 */
                                                                                               /* 0x2232929280429297 */
        /*0148*/                   DMUL R8, R4, R4;                                            /* 0x5000000010421c01 */
        /*0150*/                   DFMA R8, R20, R20, R8;                                      /* 0x2010000051421c01 */
        /*0158*/                   IADD32I R12, R9, -0x3500000;                                /* 0x0bf2c00000931c02 */
        /*0160*/                   MUFU.RSQ64H R13, R9;                                        /* 0xc80000001c935c00 */
        /*0168*/                   DMUL R6, R12, R12;                                          /* 0x5000000030c19c01 */
        /*0170*/                   DFMA R6, -R8, R6, c[0x2][0x0];                              /* 0x200c880000819e01 */
        /*0178*/                   DFMA R10, R6, 0.375, R10;                                   /* 0x2014cff600629c01 */
                                                                                               /* 0x2292804202820297 */
        /*0188*/                   DMUL R6, R12, R6;                                           /* 0x5000000018c19c01 */
        /*0190*/                   DFMA R18, R10, R6, R12;                                     /* 0x2018000018a49c01 */
        /*0198*/                   ISETP.LT.U32.AND P0, PT, R12, c[0x2][0x8], PT;              /* 0x188e480020c1dc03 */
        /*01a0*/                   DMUL R10, R8, R18;                                          /* 0x5000000048829c01 */
        /*01a8*/                   IADD32I R17, R19, -0x100000;                                /* 0x0bffc00001345c02 */
        /*01b0*/                   MOV R16, R18;                                               /* 0x2800000048041de4 */
        /*01b8*/                   DFMA R14, -R10, R10, R8;                                    /* 0x2010000028a39e01 */
                                                                                               /* 0x221200420042b207 */
        /*01c8*/                   DFMA R6, R14, R16, R10;                                     /* 0x2014000040e19c01 */
        /*01d0*/               @P0 BRA 0x220;                                                  /* 0x40000001200001e7 */
        /*01d8*/                   MOV R6, R8;                                                 /* 0x2800000020019de4 */
        /*01e0*/                   MOV R7, R9;                                                 /* 0x280000002401dde4 */
        /*01e8*/                   MOV R8, R14;                                                /* 0x2800000038021de4 */
        /*01f0*/                   MOV R13, R11;                                               /* 0x280000002c035de4 */
        /*01f8*/                   MOV R9, R15;                                                /* 0x280000003c025de4 */
                                                                                               /* 0x227202c042b2b047 */
        /*0208*/                   MOV R14, R10;                                               /* 0x2800000028039de4 */
        /*0210*/                   MOV R11, R17;                                               /* 0x280000004402dde4 */
        /*0218*/                   CAL 0x7a8;                                                  /* 0x5000001620011c07 */
        /*0220*/                   IADD R8.CC, R23, R22;                                       /* 0x4801000059721c03 */
        /*0228*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*0230*/                   IADD.X R9, R25, R0;                                         /* 0x4800000001925c43 */
        /*0238*/                   IADD R18.CC, R22, R2;                                       /* 0x4801000009649c03 */
                                                                                               /* 0x2292904280429047 */
        /*0248*/                   LD.E.64 R8, [R8+0x10];                                      /* 0x8400000040821ca5 */
        /*0250*/                   DFMA R12, R6, R6, c[0x2][0x18];                             /* 0x200c880060631c01 */
        /*0258*/                   IADD.X R19, R0, R3;                                         /* 0x480000000c04dc43 */
        /*0260*/                   MUFU.RCP64H R11, R13;                                       /* 0xc800000018d2dc00 */
        /*0268*/                   LD.E.64 R14, [R18+0x10];                                    /* 0x8400000041239ca5 */
        /*0270*/                   DFMA R2, -R12, R10, c[0x2][0x0];                            /* 0x2014880000c09e01 */
        /*0278*/                   DFMA R16, R2, R2, R2;                                       /* 0x2004000008241c01 */
                                                                                               /* 0x22820292929323f7 */
        /*0288*/                   DFMA R16, R10, R16, R10;                                    /* 0x2014000040a41c01 */
        /*0290*/                   DMUL R2, R8, c[0x2][0x10];                                  /* 0x5000480040809c01 */
        /*0298*/                   DMUL R2, R2, R14;                                           /* 0x5000000038209c01 */
        /*02a0*/                   DMUL R14, R2, R16;                                          /* 0x5000000040239c01 */
        /*02a8*/                   DFMA R10, -R12, R14, R2;                                    /* 0x2004000038c29e01 */
        /*02b0*/                   DFMA R10, R16, R10, R14;                                    /* 0x201c000029029c01 */
        /*02b8*/                   FSET.GTU.AND RZ.CC, |R3|, c[0x2][0xc], PT;                  /* 0x160f4800303fdc80 */
                                                                                               /* 0x2202b042b2b2c287 */
        /*02c8*/                   FFMA R8, RZ, R13, R11;                                      /* 0x3016000037f21c00 */
        /*02d0*/                   FSETP.GT.AND P0, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008081dc80 */
        /*02d8*/               @P0 BRA CC.NEU, 0x2f8;                                          /* 0x40000000600001a7 */
        /*02e0*/                   CAL 0x4a8;                                                  /* 0x5000000700011c07 */
        /*02e8*/                   MOV R10, R2;                                                /* 0x2800000008029de4 */
        /*02f0*/                   MOV R11, R3;                                                /* 0x280000000c02dde4 */
        /*02f8*/                   MUFU.RCP64H R3, R7;                                         /* 0xc80000001870dc00 */
                                                                                               /* 0x2292929292928047 */
        /*0308*/                   MOV32I R2, 0x1;                                             /* 0x1800000004009de2 */
        /*0310*/                   DMUL R20, R20, R10;                                         /* 0x5000000029451c01 */
        /*0318*/                   DFMA R8, -R6, R2, c[0x2][0x0];                              /* 0x2004880000621e01 */
        /*0320*/                   DFMA R8, R8, R8, R8;                                        /* 0x2010000020821c01 */
        /*0328*/                   DFMA R2, R2, R8, R2;                                        /* 0x2004000020209c01 */
        /*0330*/                   DMUL R8, R20, R2;                                           /* 0x5000000009421c01 */
        /*0338*/                   DFMA R12, -R6, R8, R20;                                     /* 0x2028000020631e01 */
                                                                                               /* 0x22b042c282804207 */
        /*0348*/                   DFMA R2, R2, R12, R8;                                       /* 0x2010000030209c01 */
        /*0350*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
        /*0358*/                   MOV R12, R6;                                                /* 0x2800000018031de4 */
        /*0360*/                   FFMA R8, RZ, R7, R3;                                        /* 0x300600001ff21c00 */
        /*0368*/                   FSETP.GT.AND P0, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008081dc80 */
        /*0370*/                   MOV R13, R7;                                                /* 0x280000001c035de4 */
        /*0378*/               @P0 BRA CC.NEU, 0x3a0;                                          /* 0x40000000800001a7 */
                                                                                               /* 0x228042c042b2b047 */
        /*0388*/                   MOV R2, R20;                                                /* 0x2800000050009de4 */
        /*0390*/                   MOV R3, R21;                                                /* 0x280000005400dde4 */
        /*0398*/                   CAL 0x4a8;                                                  /* 0x5000000420011c07 */
        /*03a0*/                   IADD R8.CC, R23, R22;                                       /* 0x4801000059721c03 */
        /*03a8*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
        /*03b0*/                   IADD.X R9, R25, R0;                                         /* 0x4800000001925c43 */
        /*03b8*/                   MUFU.RCP64H R19, R7;                                        /* 0xc80000001874dc00 */
                                                                                               /* 0x2292929250423047 */
        /*03c8*/                   LD.E.64 R12, [R8+0x4e210];                                  /* 0x8400138840831ca5 */
        /*03d0*/                   DFMA R14, -R6, R18, c[0x2][0x0];                            /* 0x2024880000639e01 */
        /*03d8*/                   LD.E.64 R20, [R8+0x5dc10];                                  /* 0x8400177040851ca5 */
        /*03e0*/                   DMUL R16, R4, R10;                                          /* 0x5000000028441c01 */
        /*03e8*/                   DFMA R14, R14, R14, R14;                                    /* 0x201c000038e39c01 */
        /*03f0*/                   DFMA R14, R18, R14, R18;                                    /* 0x2024000039239c01 */
        /*03f8*/                   DMUL R4, R14, R16;                                          /* 0x5000000040e11c01 */
                                                                                               /* 0x220293f282820297 */
        /*0408*/                   DFMA R10, -R6, R4, R16;                                     /* 0x2020000010629e01 */
        /*0410*/                   DFMA R10, R14, R10, R4;                                     /* 0x2008000028e29c01 */
        /*0418*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0xc], PT;                 /* 0x160f4800311fdc80 */
        /*0420*/                   FFMA R0, RZ, R7, R11;                                       /* 0x301600001ff01c00 */
        /*0428*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0x20], PT;                /* 0x220e48008001dc80 */
        /*0430*/                   DADD R2, R2, R12;                                           /* 0x4800000030209c01 */
        /*0438*/                   ST.E.64 [R8+0x4e210], R2;                                   /* 0x9400138840809ca5 */
                                                                                               /* 0x2202b2b042b04207 */
        /*0448*/                   MOV R12, R6;                                                /* 0x2800000018031de4 */
        /*0450*/                   MOV R13, R7;                                                /* 0x280000001c035de4 */
        /*0458*/               @P0 BRA CC.NEU, 0x490;                                          /* 0x40000000c00001a7 */
        /*0460*/                   MOV R2, R16;                                                /* 0x2800000040009de4 */
        /*0468*/                   MOV R3, R17;                                                /* 0x280000004400dde4 */
        /*0470*/                   CAL 0x4a8;                                                  /* 0x50000000c0011c07 */
        /*0478*/                   MOV R10, R2;                                                /* 0x2800000008029de4 */
                                                                                               /* 0x22720042e04292b7 */
        /*0488*/                   MOV R11, R3;                                                /* 0x280000000c02dde4 */
        /*0490*/                   DADD R10, R20, R10;                                         /* 0x4800000029429c01 */
        /*0498*/                   ST.E.64 [R8+0x5dc10], R10;                                  /* 0x9400177040829ca5 */
        /*04a0*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*04a8*/                   LOP32I.AND R14, R13, 0x40000000;                            /* 0x3900000000d39c02 */
        /*04b0*/                   LOP32I.AND R26, R3, 0x7f800000;                             /* 0x39fe000000369c02 */
        /*04b8*/                   MOV32I R24, 0x5ff00000;                                     /* 0x197fc00000061de2 */
                                                                                               /* 0x22c042028042c047 */
        /*04c8*/                   ISETP.LT.U32.AND P0, PT, R14, c[0x2][0x24], PT;             /* 0x188e480090e1dc03 */
        /*04d0*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
        /*04d8*/                   SEL R15, R24, c[0x2][0x28], P0;                             /* 0x20004800a183dc04 */
        /*04e0*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
        /*04e8*/                   DMUL R16, R12, R14;                                         /* 0x5000000038c41c01 */
        /*04f0*/                   ISETP.LT.U32.AND P0, PT, R26, c[0x2][0x2c], PT;             /* 0x188e4800b1a1dc03 */
        /*04f8*/                   MOV R28, RZ;                                                /* 0x28000000fc071de4 */
                                                                                               /* 0x2292925232928047 */
        /*0508*/                   MUFU.RCP64H R19, R17;                                       /* 0xc80000001914dc00 */
        /*0510*/                   SEL R29, R24, c[0x2][0x28], P0;                             /* 0x20004800a1875c04 */
        /*0518*/                   DFMA R30, -R16, R18, c[0x2][0x0];                           /* 0x2024880001079e01 */
        /*0520*/                   DFMA R30, R30, R30, R30;                                    /* 0x203c000079e79c01 */
        /*0528*/                   DMUL R28, R2, R28;                                          /* 0x5000000070271c01 */
        /*0530*/                   DFMA R18, R18, R30, R18;                                    /* 0x2024000079249c01 */
        /*0538*/                   DMUL R30, R28, R18;                                         /* 0x5000000049c79c01 */
                                                                                               /* 0x22b20042b2d29297 */
        /*0548*/                   DFMA R16, -R16, R30, R28;                                   /* 0x2038000079041e01 */
        /*0550*/                   DFMA R28, R18, R16, R30;                                    /* 0x203c000041271c01 */
        /*0558*/                   DSETP.LEU.AND P0, PT, |R28|, RZ, PT;                        /* 0x1d8e0000fdc1dc81 */
        /*0560*/               @P0 BRA 0x718;                                                  /* 0x40000006c00001e7 */
        /*0568*/                   ISETP.GT.U32.AND P0, PT, R26, c[0x2][0x30], PT;             /* 0x1a0e4800c1a1dc03 */
        /*0570*/                   DMUL R18, R14, R28;                                         /* 0x5000000070e49c01 */
        /*0578*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
                                                                                               /* 0x22d2925232523287 */
        /*0588*/                   SEL R17, R24, c[0x2][0x28], P0;                             /* 0x20004800a1845c04 */
        /*0590*/                   DMUL R26, R28, R16;                                         /* 0x5000000041c69c01 */
        /*0598*/                   DMUL R16, R16, R18;                                         /* 0x5000000049041c01 */
        /*05a0*/                   DMUL R14, R14, R26;                                         /* 0x5000000068e39c01 */
        /*05a8*/                   DFMA R18, R12, R16, -R2;                                    /* 0x2004000040c49d01 */
        /*05b0*/                   DFMA R28, R12, R14, -R2;                                    /* 0x2004000038c71d01 */
        /*05b8*/                   DSETP.GT.AND P0, PT, |R18|, |R28|, PT;                      /* 0x1a0e00007121dcc1 */
                                                                                               /* 0x22220042b2c27207 */
        /*05c8*/                   SEL R15, R15, R17, P0;                                      /* 0x2000000044f3dc04 */
        /*05d0*/                   SEL R14, R14, R16, P0;                                      /* 0x2000000040e39c04 */
        /*05d8*/                   FSETP.GTU.AND P0, PT, |R15|, 1.469367938527859385e-39, PT;  /* 0x260ec00400f1dc80 */
        /*05e0*/               @P0 BRA 0x790;                                                  /* 0x40000006a00001e7 */
        /*05e8*/                   FSETP.LT.AND P0, PT, |R3|, 1.5046327690525280102e-36, PT;   /* 0x208ec1000031dc80 */
        /*05f0*/                   LOP32I.AND R24, R14, 0xfffffffe;                            /* 0x3bfffffff8e61c02 */
        /*05f8*/                   MOV32I R17, 0x58500000;                                     /* 0x1961400000045de2 */
                                                                                               /* 0x2202023280428047 */
        /*0608*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*0610*/                   LOP32I.OR R27, R14, 0x1;                                    /* 0x3800000004e6dc42 */
        /*0618*/                   SEL R17, R17, c[0x2][0x4], P0;                              /* 0x2000480011145c04 */
        /*0620*/                   MOV R14, R24;                                               /* 0x2800000060039de4 */
        /*0628*/                   DMUL R12, R12, R16;                                         /* 0x5000000040c31c01 */
        /*0630*/                   DMUL R2, R2, R16;                                           /* 0x5000000040209c01 */
        /*0638*/                   MOV R16, R27;                                               /* 0x280000006c041de4 */
                                                                                               /* 0x22c04282d2923277 */
        /*0648*/                   MOV R17, R15;                                               /* 0x280000003c045de4 */
        /*0650*/                   DFMA R18, R14, R12, -R2;                                    /* 0x2004000030e49d01 */
        /*0658*/                   DFMA R16, R16, R12, -R2;                                    /* 0x2004000031041d01 */
        /*0660*/                   DSETP.GT.AND P0, PT, |R18|, |R16|, PT;                      /* 0x1a0e00004121dcc1 */
        /*0668*/                   SEL R14, R27, R24, P0;                                      /* 0x2000000061b39c04 */
        /*0670*/                   IADD32I R19.CC, R14, 0x1;                                   /* 0x0c00000004e4dc02 */
        /*0678*/                   LOP32I.AND R17, R14, 0x1;                                   /* 0x3800000004e45c02 */
                                                                                               /* 0x2262020042c20047 */
        /*0688*/                   ISETP.EQ.U32.AND P0, PT, R17, 0x1, PT;                      /* 0x190ec0000511dc03 */
        /*0690*/                   IADD.X R18, RZ, R15;                                        /* 0x480000003ff49c43 */
        /*0698*/                   IADD32I R16.CC, R14, -0x1;                                  /* 0x0ffffffffce41c02 */
        /*06a0*/                   SEL R19, R14, R19, !P0;                                     /* 0x201000004ce4dc04 */
        /*06a8*/                   IADD32I.X R17, R15, -0x1;                                   /* 0x0bfffffffcf45c42 */
        /*06b0*/                   SEL R18, R15, R18, !P0;                                     /* 0x2010000048f49c04 */
        /*06b8*/                   SEL R16, R16, R14, !P0;                                     /* 0x2010000039041c04 */
                                                                                               /* 0x2202d29232820207 */
        /*06c8*/                   SEL R17, R17, R15, !P0;                                     /* 0x201000003d145c04 */
        /*06d0*/                   MOV R14, R19;                                               /* 0x280000004c039de4 */
        /*06d8*/                   MOV R15, R18;                                               /* 0x280000004803dde4 */
        /*06e0*/                   DFMA R14, R12, R14, -R2;                                    /* 0x2004000038c39d01 */
        /*06e8*/                   DFMA R2, R12, R16, -R2;                                     /* 0x2004000040c09d01 */
        /*06f0*/                   DSETP.GT.AND P0, PT, |R14|, |R2|, PT;                       /* 0x1a0e000008e1dcc1 */
        /*06f8*/                   SEL R14, R16, R19, P0;                                      /* 0x200000004d039c04 */
                                                                                               /* 0x22d28042b2d2e047 */
        /*0708*/                   SEL R15, R17, R18, P0;                                      /* 0x200000004913dc04 */
        /*0710*/                   BRA 0x790;                                                  /* 0x40000001e0001de7 */
        /*0718*/                   DSETP.EQ.AND P0, PT, R28, RZ, PT;                           /* 0x190e0000fdc1dc01 */
        /*0720*/               @P0 BRA 0x780;                                                  /* 0x40000001600001e7 */
        /*0728*/                   MUFU.RCP64H R15, R13;                                       /* 0xc800000018d3dc00 */
        /*0730*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
        /*0738*/                   DSETP.GT.AND P1, PT, |R14|, RZ, PT;                         /* 0x1a0e0000fce3dc81 */
                                                                                               /* 0x22e20282027202d7 */
        /*0748*/              @!P1 DSETP.NEU.AND P0, PT, |R12|, +INF , PT;                     /* 0x1e8edffc00c1e481 */
        /*0750*/              @!P1 SEL R12, R12, R14, P0;                                      /* 0x2000000038c32404 */
        /*0758*/              @!P1 SEL R13, R13, R15, P0;                                      /* 0x200000003cd36404 */
        /*0760*/              @!P1 MOV R14, R12;                                               /* 0x280000003003a5e4 */
        /*0768*/              @!P1 MOV R15, R13;                                               /* 0x280000003403e5e4 */
        /*0770*/                   DMUL R14, R2, R14;                                          /* 0x5000000038239c01 */
        /*0778*/                   BRA 0x790;                                                  /* 0x4000000040001de7 */
                                                                                               /* 0x22b20042e20042b7 */
        /*0788*/                   DMUL R14, R2, R12;                                          /* 0x5000000030239c01 */
        /*0790*/                   MOV R2, R14;                                                /* 0x2800000038009de4 */
        /*0798*/                   MOV R3, R15;                                                /* 0x280000003c00dde4 */
        /*07a0*/                   RET;                                                        /* 0x9000000000001de7 */
        /*07a8*/                   ISETP.LT.U32.AND P0, PT, R12, c[0x2][0x34], PT;             /* 0x188e4800d0c1dc03 */
        /*07b0*/                   MOV R10, R18;                                               /* 0x2800000048029de4 */
        /*07b8*/                   MOV R12, R14;                                               /* 0x2800000038031de4 */
                                                                                               /* 0x2202d29282c292b7 */
        /*07c8*/               @P0 BRA 0x818;                                                  /* 0x40000001200001e7 */
        /*07d0*/                   DFMA.RM R8, R8, R10, R12;                                   /* 0x2098000028821c01 */
        /*07d8*/                   IADD32I R10.CC, R8, 0x1;                                    /* 0x0c00000004829c02 */
        /*07e0*/                   IADD.X R11, RZ, R9;                                         /* 0x4800000027f2dc43 */
        /*07e8*/                   DFMA.RP R6, -R8, R10, R6;                                   /* 0x210c000028819e01 */
        /*07f0*/                   DSETP.GT.AND P0, PT, R6, RZ, PT;                            /* 0x1a0e0000fc61dc01 */
        /*07f8*/                   SEL R8, R10, R8, P0;                                        /* 0x2000000020a21c04 */
                                                                                               /* 0x220042c2b2d2e047 */
        /*0808*/                   SEL R9, R11, R9, P0;                                        /* 0x2000000024b25c04 */
        /*0810*/                   BRA 0x8f8;                                                  /* 0x4000000380001de7 */
        /*0818*/                   DSETP.EQU.AND P0, PT, R6, RZ, PT;                           /* 0x1d0e0000fc61dc01 */
        /*0820*/               @P0 BRA 0x8f0;                                                  /* 0x40000003200001e7 */
        /*0828*/                   ISETP.LT.AND P0, PT, R7, RZ, PT;                            /* 0x188e0000fc71dc23 */
        /*0830*/               @P0 MOV32I R8, 0x0;                                             /* 0x18000000000201e2 */
        /*0838*/               @P0 MOV32I R9, 0xfff80000;                                      /* 0x1bffe000000241e2 */
                                                                                               /* 0x2282804202b2c2b7 */
        /*0848*/               @P0 BRA 0x8f8;                                                  /* 0x40000002a00001e7 */
        /*0850*/                   ISETP.GT.AND P0, PT, R7, c[0x2][0x38], PT;                  /* 0x1a0e4800e071dc23 */
        /*0858*/               @P0 BRA 0x8f0;                                                  /* 0x40000002400001e7 */
        /*0860*/                   DMUL R6, R6, 8.11296384146066816958e+31;                    /* 0x5000d1a400619c01 */
        /*0868*/                   MOV R8, RZ;                                                 /* 0x28000000fc021de4 */
        /*0870*/                   MOV32I R12, 0x0;                                            /* 0x1800000000031de2 */
        /*0878*/                   MUFU.RSQ64H R9, R7;                                         /* 0xc80000001c725c00 */
                                                                                               /* 0x2202929232928207 */
        /*0888*/                   DMUL R10, R8, R8;                                           /* 0x5000000020829c01 */
        /*0890*/                   MOV32I R13, 0x3fe00000;                                     /* 0x18ff800000035de2 */
        /*0898*/                   DFMA R10, -R6, R10, c[0x2][0x0];                            /* 0x2014880000629e01 */
        /*08a0*/                   DFMA R12, R10, 0.375, R12;                                  /* 0x2018cff600a31c01 */
        /*08a8*/                   DMUL R10, R8, R10;                                          /* 0x5000000028829c01 */
        /*08b0*/                   DFMA R8, R12, R10, R8;                                      /* 0x2010000028c21c01 */
        /*08b8*/                   DMUL R10, R6, R8;                                           /* 0x5000000020629c01 */
                                                                                               /* 0x2202b2e042929287 */
        /*08c8*/                   IADD32I R9, R9, -0x100000;                                  /* 0x0bffc00000925c02 */
        /*08d0*/                   DFMA R6, -R10, R10, R6;                                     /* 0x200c000028a19e01 */
        /*08d8*/                   DFMA R8, R8, R6, R10;                                       /* 0x2014000018821c01 */
        /*08e0*/                   IADD32I R9, R9, -0x3500000;                                 /* 0x0bf2c00000925c02 */
        /*08e8*/                   BRA 0x8f8;                                                  /* 0x4000000020001de7 */
        /*08f0*/                   DADD R8, R6, R6;                                            /* 0x4800000018621c01 */
        /*08f8*/                   MOV R6, R8;                                                 /* 0x2800000020019de4 */
                                                                                               /* 0x200000000002e047 */
        /*0908*/                   MOV R7, R9;                                                 /* 0x280000002401dde4 */
        /*0910*/                   RET;                                                        /* 0x9000000000001de7 */
        /*0918*/                   BRA 0x918;                                                  /* 0x4003ffffe0001de7 */
        /*0920*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*0928*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*0930*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*0938*/                   NOP;                                                        /* 0x4000000000001de4 */
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
code version = [6,1]
producer = cuda
host = linux
compile_size = 64bit
compressed








.version 6.1
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
add.s64 %rd4, %rd3, %rd2;
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
.reg .b64 %rd<15>;


ld.param.u64 %rd1, [_Z17codegen_add_forceP4BodyS0__param_0];
ld.param.u64 %rd2, [_Z17codegen_add_forceP4BodyS0__param_1];
cvta.to.global.u64 %rd3, %rd2;
cvta.to.global.u64 %rd4, %rd1;
setp.eq.s64	%p1, %rd4, %rd3;
@%p1 bra BB1_2;

shl.b64 %rd5, %rd2, 3;
mov.u64 %rd6, __Bodydata_buffer;
add.s64 %rd7, %rd6, 64016;
add.s64 %rd8, %rd7, %rd5;
shl.b64 %rd9, %rd1, 3;
add.s64 %rd10, %rd7, %rd9;
ld.global.f64 %fd1, [%rd10];
ld.global.f64 %fd2, [%rd8];
sub.f64 %fd3, %fd2, %fd1;
ld.global.f64 %fd4, [%rd10+64000];
ld.global.f64 %fd5, [%rd8+64000];
sub.f64 %fd6, %fd5, %fd4;
mul.f64 %fd7, %fd6, %fd6;
fma.rn.f64 %fd8, %fd3, %fd3, %fd7;
sqrt.rn.f64 %fd9, %fd8;
add.s64 %rd11, %rd6, 16;
add.s64 %rd12, %rd11, %rd9;
ld.global.f64 %fd10, [%rd12];
mul.f64 %fd11, %fd10, 0d3DD257B4D002790D;
add.s64 %rd13, %rd11, %rd5;
ld.global.f64 %fd12, [%rd13];
mul.f64 %fd13, %fd11, %fd12;
fma.rn.f64 %fd14, %fd9, %fd9, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd15, %fd13, %fd14;
mul.f64 %fd16, %fd3, %fd15;
div.rn.f64 %fd17, %fd16, %fd9;
add.s64 %rd14, %rd6, %rd9;
ld.global.f64 %fd18, [%rd14+320016];
add.f64 %fd19, %fd17, %fd18;
ld.global.f64 %fd20, [%rd14+384016];
st.global.f64 [%rd14+320016], %fd19;
mul.f64 %fd21, %fd6, %fd15;
div.rn.f64 %fd22, %fd21, %fd9;
add.f64 %fd23, %fd22, %fd20;
st.global.f64 [%rd14+384016], %fd23;

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
add.s64 %rd4, %rd3, %rd2;
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
add.s64 %rd4, %rd3, %rd2;
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
add.s64 %rd1, %rd8, 320016;
add.s64 %rd2, %rd8, 16;
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


.visible .entry _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_11reset_forceEvEEE4callEvEUlvE_JEEEvT_DpT0_(
.param .align 1 .b8 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_11reset_forceEvEEE4callEvEUlvE_JEEEvT_DpT0__param_0[1]
)
{
.reg .pred %p<2>;
.reg .b32 %r<5>;
.reg .b64 %rd<7>;


mov.u32 %r1, %tid.x;
mov.u32 %r2, %ntid.x;
mov.u32 %r3, %ctaid.x;
mad.lo.s32 %r4, %r2, %r3, %r1;
cvt.s64.s32	%rd1, %r4;
ld.global.u64 %rd2, [__Bodydata_buffer];
setp.ge.u64	%p1, %rd1, %rd2;
@%p1 bra BB7_2;

shl.b64 %rd3, %rd1, 3;
mov.u64 %rd4, __Bodydata_buffer;
add.s64 %rd5, %rd3, %rd4;
mov.u64 %rd6, 0;
st.global.u64 [%rd5+320024], %rd6;
st.global.u64 [%rd5+384024], %rd6;

BB7_2:
ret;
}


.visible .entry _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_22add_all_forces_to_thisEvEEE4callEvEUlvE_JEEEvT_DpT0_(
.param .align 1 .b8 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_22add_all_forces_to_thisEvEEE4callEvEUlvE_JEEEvT_DpT0__param_0[1]
)
{
.reg .pred %p<8>;
.reg .b32 %r<25>;
.reg .f64 %fd<116>;
.reg .b64 %rd<61>;


mov.u32 %r1, %tid.x;
mov.u32 %r2, %ntid.x;
mov.u32 %r3, %ctaid.x;
mad.lo.s32 %r4, %r2, %r3, %r1;
cvt.s64.s32	%rd1, %r4;
ld.global.u64 %rd19, [__Bodydata_buffer];
setp.ge.u64	%p1, %rd1, %rd19;
@%p1 bra BB8_13;

add.s64 %rd2, %rd1, 1;
shl.b64 %rd27, %rd1, 3;
mov.u64 %rd59, __Bodydata_buffer;
add.s64 %rd28, %rd27, %rd59;
add.s64 %rd3, %rd28, 64024;
add.s64 %rd4, %rd28, 320024;
mov.u64 %rd60, 8000;
mov.u64 %rd58, 5;
mov.u64 %rd57, 4;
mov.u64 %rd56, 3;
mov.u64 %rd55, 2;
mov.u64 %rd54, 1;

BB8_2:
setp.eq.s64	%p2, %rd54, %rd2;
@%p2 bra BB8_4;

shl.b64 %rd29, %rd54, 3;
mov.u64 %rd30, __Bodydata_buffer;
add.s64 %rd31, %rd30, %rd29;
ld.global.f64 %fd1, [%rd3];
ld.global.f64 %fd2, [%rd31+64016];
sub.f64 %fd3, %fd2, %fd1;
ld.global.f64 %fd4, [%rd3+64000];
ld.global.f64 %fd5, [%rd31+128016];
sub.f64 %fd6, %fd5, %fd4;
mul.f64 %fd7, %fd6, %fd6;
fma.rn.f64 %fd8, %fd3, %fd3, %fd7;
sqrt.rn.f64 %fd9, %fd8;
mul.wide.s32 %rd32, %r4, 8;
add.s64 %rd33, %rd30, %rd32;
ld.global.f64 %fd10, [%rd33+24];
mul.f64 %fd11, %fd10, 0d3DD257B4D002790D;
ld.global.f64 %fd12, [%rd59+24];
mul.f64 %fd13, %fd11, %fd12;
fma.rn.f64 %fd14, %fd9, %fd9, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd15, %fd13, %fd14;
mul.f64 %fd16, %fd3, %fd15;
div.rn.f64 %fd17, %fd16, %fd9;
ld.global.f64 %fd18, [%rd4];
add.f64 %fd19, %fd17, %fd18;
ld.global.f64 %fd20, [%rd4+64000];
st.global.f64 [%rd4], %fd19;
mul.f64 %fd21, %fd6, %fd15;
div.rn.f64 %fd22, %fd21, %fd9;
add.f64 %fd23, %fd22, %fd20;
st.global.f64 [%rd4+64000], %fd23;

BB8_4:
setp.eq.s64	%p3, %rd55, %rd2;
@%p3 bra BB8_6;

shl.b64 %rd34, %rd55, 3;
mov.u64 %rd35, __Bodydata_buffer;
add.s64 %rd36, %rd35, %rd34;
ld.global.f64 %fd24, [%rd3];
ld.global.f64 %fd25, [%rd36+64016];
sub.f64 %fd26, %fd25, %fd24;
ld.global.f64 %fd27, [%rd3+64000];
ld.global.f64 %fd28, [%rd36+128016];
sub.f64 %fd29, %fd28, %fd27;
mul.f64 %fd30, %fd29, %fd29;
fma.rn.f64 %fd31, %fd26, %fd26, %fd30;
sqrt.rn.f64 %fd32, %fd31;
mul.wide.s32 %rd37, %r4, 8;
add.s64 %rd38, %rd35, %rd37;
ld.global.f64 %fd33, [%rd38+24];
mul.f64 %fd34, %fd33, 0d3DD257B4D002790D;
ld.global.f64 %fd35, [%rd59+32];
mul.f64 %fd36, %fd34, %fd35;
fma.rn.f64 %fd37, %fd32, %fd32, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd38, %fd36, %fd37;
mul.f64 %fd39, %fd26, %fd38;
div.rn.f64 %fd40, %fd39, %fd32;
ld.global.f64 %fd41, [%rd4];
add.f64 %fd42, %fd40, %fd41;
ld.global.f64 %fd43, [%rd4+64000];
st.global.f64 [%rd4], %fd42;
mul.f64 %fd44, %fd29, %fd38;
div.rn.f64 %fd45, %fd44, %fd32;
add.f64 %fd46, %fd45, %fd43;
st.global.f64 [%rd4+64000], %fd46;

BB8_6:
setp.eq.s64	%p4, %rd56, %rd2;
@%p4 bra BB8_8;

shl.b64 %rd39, %rd56, 3;
mov.u64 %rd40, __Bodydata_buffer;
add.s64 %rd41, %rd40, %rd39;
ld.global.f64 %fd47, [%rd3];
ld.global.f64 %fd48, [%rd41+64016];
sub.f64 %fd49, %fd48, %fd47;
ld.global.f64 %fd50, [%rd3+64000];
ld.global.f64 %fd51, [%rd41+128016];
sub.f64 %fd52, %fd51, %fd50;
mul.f64 %fd53, %fd52, %fd52;
fma.rn.f64 %fd54, %fd49, %fd49, %fd53;
sqrt.rn.f64 %fd55, %fd54;
mul.wide.s32 %rd42, %r4, 8;
add.s64 %rd43, %rd40, %rd42;
ld.global.f64 %fd56, [%rd43+24];
mul.f64 %fd57, %fd56, 0d3DD257B4D002790D;
ld.global.f64 %fd58, [%rd59+40];
mul.f64 %fd59, %fd57, %fd58;
fma.rn.f64 %fd60, %fd55, %fd55, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd61, %fd59, %fd60;
mul.f64 %fd62, %fd49, %fd61;
div.rn.f64 %fd63, %fd62, %fd55;
ld.global.f64 %fd64, [%rd4];
add.f64 %fd65, %fd63, %fd64;
ld.global.f64 %fd66, [%rd4+64000];
st.global.f64 [%rd4], %fd65;
mul.f64 %fd67, %fd52, %fd61;
div.rn.f64 %fd68, %fd67, %fd55;
add.f64 %fd69, %fd68, %fd66;
st.global.f64 [%rd4+64000], %fd69;

BB8_8:
setp.eq.s64	%p5, %rd57, %rd2;
@%p5 bra BB8_10;

shl.b64 %rd44, %rd57, 3;
mov.u64 %rd45, __Bodydata_buffer;
add.s64 %rd46, %rd45, %rd44;
ld.global.f64 %fd70, [%rd3];
ld.global.f64 %fd71, [%rd46+64016];
sub.f64 %fd72, %fd71, %fd70;
ld.global.f64 %fd73, [%rd3+64000];
ld.global.f64 %fd74, [%rd46+128016];
sub.f64 %fd75, %fd74, %fd73;
mul.f64 %fd76, %fd75, %fd75;
fma.rn.f64 %fd77, %fd72, %fd72, %fd76;
sqrt.rn.f64 %fd78, %fd77;
mul.wide.s32 %rd47, %r4, 8;
add.s64 %rd48, %rd45, %rd47;
ld.global.f64 %fd79, [%rd48+24];
mul.f64 %fd80, %fd79, 0d3DD257B4D002790D;
ld.global.f64 %fd81, [%rd59+48];
mul.f64 %fd82, %fd80, %fd81;
fma.rn.f64 %fd83, %fd78, %fd78, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd84, %fd82, %fd83;
mul.f64 %fd85, %fd72, %fd84;
div.rn.f64 %fd86, %fd85, %fd78;
ld.global.f64 %fd87, [%rd4];
add.f64 %fd88, %fd86, %fd87;
ld.global.f64 %fd89, [%rd4+64000];
st.global.f64 [%rd4], %fd88;
mul.f64 %fd90, %fd75, %fd84;
div.rn.f64 %fd91, %fd90, %fd78;
add.f64 %fd92, %fd91, %fd89;
st.global.f64 [%rd4+64000], %fd92;

BB8_10:
setp.eq.s64	%p6, %rd58, %rd2;
@%p6 bra BB8_12;

shl.b64 %rd49, %rd58, 3;
mov.u64 %rd50, __Bodydata_buffer;
add.s64 %rd51, %rd50, %rd49;
ld.global.f64 %fd93, [%rd3];
ld.global.f64 %fd94, [%rd51+64016];
sub.f64 %fd95, %fd94, %fd93;
ld.global.f64 %fd96, [%rd3+64000];
ld.global.f64 %fd97, [%rd51+128016];
sub.f64 %fd98, %fd97, %fd96;
mul.f64 %fd99, %fd98, %fd98;
fma.rn.f64 %fd100, %fd95, %fd95, %fd99;
sqrt.rn.f64 %fd101, %fd100;
mul.wide.s32 %rd52, %r4, 8;
add.s64 %rd53, %rd50, %rd52;
ld.global.f64 %fd102, [%rd53+24];
mul.f64 %fd103, %fd102, 0d3DD257B4D002790D;
ld.global.f64 %fd104, [%rd59+56];
mul.f64 %fd105, %fd103, %fd104;
fma.rn.f64 %fd106, %fd101, %fd101, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd107, %fd105, %fd106;
mul.f64 %fd108, %fd95, %fd107;
div.rn.f64 %fd109, %fd108, %fd101;
ld.global.f64 %fd110, [%rd4];
add.f64 %fd111, %fd109, %fd110;
ld.global.f64 %fd112, [%rd4+64000];
st.global.f64 [%rd4], %fd111;
mul.f64 %fd113, %fd98, %fd107;
div.rn.f64 %fd114, %fd113, %fd101;
add.f64 %fd115, %fd114, %fd112;
st.global.f64 [%rd4+64000], %fd115;

BB8_12:
add.s64 %rd60, %rd60, -5;
add.s64 %rd59, %rd59, 40;
add.s64 %rd58, %rd58, 5;
add.s64 %rd57, %rd57, 5;
add.s64 %rd56, %rd56, 5;
add.s64 %rd55, %rd55, 5;
add.s64 %rd54, %rd54, 5;
setp.ne.s64	%p7, %rd60, 0;
@%p7 bra BB8_2;

BB8_13:
ret;
}


.visible .entry _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEdEUldE_JdEEEvT_DpT0_(
.param .align 1 .b8 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEdEUldE_JdEEEvT_DpT0__param_0[1],
.param .f64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEdEUldE_JdEEEvT_DpT0__param_1
)
{
.reg .pred %p<8>;
.reg .b32 %r<5>;
.reg .f64 %fd<25>;
.reg .b64 %rd<8>;


ld.param.f64 %fd4, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEdEUldE_JdEEEvT_DpT0__param_1];
mov.u32 %r1, %tid.x;
mov.u32 %r2, %ntid.x;
mov.u32 %r3, %ctaid.x;
mad.lo.s32 %r4, %r2, %r3, %r1;
cvt.s64.s32	%rd1, %r4;
ld.global.u64 %rd4, [__Bodydata_buffer];
setp.ge.u64	%p1, %rd1, %rd4;
@%p1 bra BB9_5;

shl.b64 %rd5, %rd1, 3;
mov.u64 %rd6, __Bodydata_buffer;
add.s64 %rd7, %rd6, %rd5;
ld.global.f64 %fd5, [%rd7+320024];
mul.f64 %fd6, %fd5, %fd4;
ld.global.f64 %fd7, [%rd7+24];
div.rn.f64 %fd8, %fd6, %fd7;
add.s64 %rd2, %rd7, 192024;
ld.global.f64 %fd9, [%rd7+192024];
add.f64 %fd10, %fd8, %fd9;
ld.global.f64 %fd11, [%rd7+384024];
ld.global.f64 %fd12, [%rd7+256024];
ld.global.f64 %fd13, [%rd7+64024];
ld.global.f64 %fd14, [%rd7+128024];
st.global.f64 [%rd7+192024], %fd10;
mul.f64 %fd15, %fd11, %fd4;
fma.rn.f64 %fd16, %fd10, %fd4, %fd13;
st.global.f64 [%rd7+64024], %fd16;
ld.global.f64 %fd17, [%rd7+24];
div.rn.f64 %fd18, %fd15, %fd17;
add.f64 %fd19, %fd18, %fd12;
st.global.f64 [%rd7+256024], %fd19;
add.s64 %rd3, %rd7, 64024;
fma.rn.f64 %fd24, %fd19, %fd4, %fd14;
st.global.f64 [%rd7+128024], %fd24;
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


.visible .entry _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEdEUldE_JdEEEvT_DpT0_(
.param .align 1 .b8 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEdEUldE_JdEEEvT_DpT0__param_0[1],
.param .f64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEdEUldE_JdEEEvT_DpT0__param_1
)
{
.reg .pred %p<3>;
.reg .b32 %r<9>;
.reg .f64 %fd<87>;
.reg .b64 %rd<9>;


ld.param.f64 %fd1, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEdEUldE_JdEEEvT_DpT0__param_1];
mov.u32 %r3, %tid.x;
mov.u32 %r4, %ntid.x;
mov.u32 %r5, %ctaid.x;
mad.lo.s32 %r6, %r4, %r5, %r3;
cvt.s64.s32	%rd1, %r6;
ld.global.u64 %rd6, [__Bodydata_buffer];
setp.ge.u64	%p1, %rd1, %rd6;
@%p1 bra BB10_3;

shl.b64 %rd7, %rd1, 3;
mov.u64 %rd8, __Bodydata_buffer;
add.s64 %rd3, %rd8, %rd7;
add.s64 %rd2, %rd3, 320024;
add.s64 %rd4, %rd3, 192024;
add.s64 %rd5, %rd3, 64024;
mov.u32 %r8, -1000;

BB10_2:
ld.global.f64 %fd2, [%rd2];
mul.f64 %fd3, %fd2, %fd1;
ld.global.f64 %fd4, [%rd3+24];
div.rn.f64 %fd5, %fd3, %fd4;
ld.global.f64 %fd6, [%rd4];
add.f64 %fd7, %fd5, %fd6;
ld.global.f64 %fd8, [%rd4+64000];
st.global.f64 [%rd4], %fd7;
ld.global.f64 %fd9, [%rd2+64000];
mul.f64 %fd10, %fd9, %fd1;
ld.global.f64 %fd11, [%rd3+24];
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
ld.global.f64 %fd21, [%rd3+24];
div.rn.f64 %fd22, %fd20, %fd21;
ld.global.f64 %fd23, [%rd4];
add.f64 %fd24, %fd22, %fd23;
ld.global.f64 %fd25, [%rd4+64000];
st.global.f64 [%rd4], %fd24;
ld.global.f64 %fd26, [%rd2+64000];
mul.f64 %fd27, %fd26, %fd1;
ld.global.f64 %fd28, [%rd3+24];
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
ld.global.f64 %fd38, [%rd3+24];
div.rn.f64 %fd39, %fd37, %fd38;
ld.global.f64 %fd40, [%rd4];
add.f64 %fd41, %fd39, %fd40;
ld.global.f64 %fd42, [%rd4+64000];
st.global.f64 [%rd4], %fd41;
ld.global.f64 %fd43, [%rd2+64000];
mul.f64 %fd44, %fd43, %fd1;
ld.global.f64 %fd45, [%rd3+24];
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
ld.global.f64 %fd55, [%rd3+24];
div.rn.f64 %fd56, %fd54, %fd55;
ld.global.f64 %fd57, [%rd4];
add.f64 %fd58, %fd56, %fd57;
ld.global.f64 %fd59, [%rd4+64000];
st.global.f64 [%rd4], %fd58;
ld.global.f64 %fd60, [%rd2+64000];
mul.f64 %fd61, %fd60, %fd1;
ld.global.f64 %fd62, [%rd3+24];
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
ld.global.f64 %fd72, [%rd3+24];
div.rn.f64 %fd73, %fd71, %fd72;
ld.global.f64 %fd74, [%rd4];
add.f64 %fd75, %fd73, %fd74;
ld.global.f64 %fd76, [%rd4+64000];
st.global.f64 [%rd4], %fd75;
ld.global.f64 %fd77, [%rd2+64000];
mul.f64 %fd78, %fd77, %fd1;
ld.global.f64 %fd79, [%rd3+24];
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


