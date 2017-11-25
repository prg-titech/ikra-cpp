
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
  REG:22 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:36 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEdEUldE_JdEEEvT_DpT0_:
  REG:32 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:36 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_22add_all_forces_to_thisEvEEE4callEvEUlvE_JEEEvT_DpT0_:
  REG:46 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:321 CONSTANT[2]:64 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_11reset_forceEvEEE4callEvEUlvE_JEEEvT_DpT0_:
  REG:8 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:321 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z22codengen_simple_updateP4Bodyd:
  REG:22 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:36 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z18codegen_reset_massP4Body:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z24codegen_reset_force_halfP4Body:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z26codegen_reset_force_manualP4Body:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z14codegen_updateP4Bodyd:
  REG:32 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:36 TEXTURE:0 SURFACE:0 SAMPLER:0
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
                                                                                              /* 0x2200423042b04277 */
        /*00c8*/                   MOV R2, R4;                                                /* 0x2800000010009de4 */
        /*00d0*/                   IADD.X R3, R0, R6;                                         /* 0x480000001800dc43 */
        /*00d8*/                   MOV32I R0, 0xfffffc18;                                     /* 0x1bfffff060001de2 */
        /*00e0*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
        /*00e8*/                   IADD32I R0, R0, 0x5;                                       /* 0x0800000014001c02 */
        /*00f0*/                   LD.E.64 R4, [R2+0x4e218];                                  /* 0x8400138860211ca5 */
        /*00f8*/                   MOV32I R8, 0x1;                                            /* 0x1800000004021de2 */
                                                                                              /* 0x229252329283f237 */
        /*0108*/                   SSY 0x198;                                                 /* 0x6000000220001c07 */
        /*0110*/                   ISETP.NE.AND P0, PT, R0, RZ, PT;                           /* 0x1a8e0000fc01dc23 */
        /*0118*/                   MUFU.RCP64H R9, R7;                                        /* 0xc800000018725c00 */
        /*0120*/                   DFMA R10, -R6, R8, c[0x2][0x0];                            /* 0x2010880000629e01 */
        /*0128*/                   DFMA R10, R10, R10, R10;                                   /* 0x2014000028a29c01 */
        /*0130*/                   DMUL R4, R4, c[0x0][0x148];                                /* 0x5000400520411c01 */
        /*0138*/                   DFMA R8, R8, R10, R8;                                      /* 0x2010000028821c01 */
                                                                                              /* 0x22b2c28282029297 */
        /*0148*/                   DMUL R10, R4, R8;                                          /* 0x5000000020429c01 */
        /*0150*/                   DFMA R12, -R6, R10, R4;                                    /* 0x2008000028631e01 */
        /*0158*/                   DFMA R8, R8, R12, R10;                                     /* 0x2014000030821c01 */
        /*0160*/                   FSET.GTU.AND RZ.CC, |R5|, c[0x2][0x8], PT;                 /* 0x160f4800205fdc80 */
        /*0168*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*0170*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*0178*/               @P1 BRA CC.NEU, 0x190;                                         /* 0x40000000400005a7 */
                                                                                              /* 0x23f230423042b2b7 */
        /*0188*/                   CAL 0xaf0;                                                 /* 0x5000002580011c07 */
        /*0190*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0198*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
        /*01a0*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*01a8*/                   LD.E.64 R10, [R2+0x5dc18];                                 /* 0x8400177060229ca5 */
        /*01b0*/                   SSY 0x278;                                                 /* 0x6000000300001c07 */
        /*01b8*/                   LD.E.64 R4, [R2+0x2ee18];                                  /* 0x84000bb860211ca5 */
                                                                                              /* 0x2292929252329287 */
        /*01c8*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*01d0*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*01d8*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
        /*01e0*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*01e8*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*01f0*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*01f8*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
                                                                                              /* 0x22b2c27200429237 */
        /*0208*/                   DFMA R18, R12, R14, R16;                                   /* 0x2020000038c49c01 */
        /*0210*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
        /*0218*/                   FFMA R8, RZ, R7, R19;                                      /* 0x302600001ff21c00 */
        /*0220*/                   ST.E.64 [R2+0x2ee18], R4;                                  /* 0x94000bb860211ca5 */
        /*0228*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*0230*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003083dc80 */
        /*0238*/               @P1 BRA CC.NEU, 0x270;                                         /* 0x40000000c00005a7 */
                                                                                              /* 0x2232b2b042b2b047 */
        /*0248*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*0250*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*0258*/                   CAL 0xaf0;                                                 /* 0x5000002240011c07 */
        /*0260*/                   MOV R18, R8;                                               /* 0x2800000020049de4 */
        /*0268*/                   MOV R19, R9;                                               /* 0x280000002404dde4 */
        /*0270*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0278*/                   LD.E.64 R12, [R2+0x3e818];                                 /* 0x84000fa060231ca5 */
                                                                                              /* 0x22e203f232323047 */
        /*0288*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
        /*0290*/                   SSY 0x398;                                                 /* 0x6000000400001c07 */
        /*0298*/                   LD.E.64 R16, [R2+0x4e218];                                 /* 0x8400138860241ca5 */
        /*02a0*/                   LD.E.64 R4, [R2+0x2ee18];                                  /* 0x84000bb860211ca5 */
        /*02a8*/                   LD.E.64 R8, [R2+0xfa18];                                   /* 0x840003e860221ca5 */
        /*02b0*/                   DADD R14, R18, R12;                                        /* 0x4800000031239c01 */
        /*02b8*/                   LD.E.64 R12, [R2+0x1f418];                                 /* 0x840007d060231ca5 */
                                                                                              /* 0x229280423042e047 */
        /*02c8*/                   ST.E.64 [R2+0x3e818], R14;                                 /* 0x94000fa060239ca5 */
        /*02d0*/                   MUFU.RCP64H R19, R7;                                       /* 0xc80000001874dc00 */
        /*02d8*/                   LD.E.64 R10, [R2+0x3e818];                                 /* 0x84000fa060229ca5 */
        /*02e0*/                   DMUL R16, R16, c[0x0][0x148];                              /* 0x5000400521041c01 */
        /*02e8*/                   MOV32I R18, 0x1;                                           /* 0x1800000004049de2 */
        /*02f0*/                   DFMA R4, R4, c[0x0][0x148], R8;                            /* 0x2010400520411c01 */
        /*02f8*/                   DFMA R20, -R6, R18, c[0x2][0x0];                           /* 0x2024880000651e01 */
                                                                                              /* 0x23f0420292929297 */
        /*0308*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*0310*/                   DFMA R18, R18, R20, R18;                                   /* 0x2024000051249c01 */
        /*0318*/                   DMUL R20, R16, R18;                                        /* 0x5000000049051c01 */
        /*0320*/                   DFMA R8, -R6, R20, R16;                                    /* 0x2020000050621e01 */
        /*0328*/                   DFMA R8, R18, R8, R20;                                     /* 0x2028000021221c01 */
        /*0330*/                   ST.E.64 [R2+0xfa18], R4;                                   /* 0x940003e860211ca5 */
        /*0338*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                /* 0x160f4800211fdc80 */
                                                                                              /* 0x22b042b2c0428207 */
        /*0348*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*0350*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*0358*/                   ST.E.64 [R2+0x1f418], R12;                                 /* 0x940007d060231ca5 */
        /*0360*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*0368*/               @P1 BRA CC.NEU, 0x390;                                         /* 0x40000000800005a7 */
        /*0370*/                   MOV R4, R16;                                               /* 0x2800000040011de4 */
        /*0378*/                   MOV R5, R17;                                               /* 0x2800000044015de4 */
                                                                                              /* 0x23f230423042b2b7 */
        /*0388*/                   CAL 0xaf0;                                                 /* 0x5000001d80011c07 */
        /*0390*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0398*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
        /*03a0*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*03a8*/                   LD.E.64 R10, [R2+0x5dc18];                                 /* 0x8400177060229ca5 */
        /*03b0*/                   SSY 0x478;                                                 /* 0x6000000300001c07 */
        /*03b8*/                   LD.E.64 R4, [R2+0x2ee18];                                  /* 0x84000bb860211ca5 */
                                                                                              /* 0x2292929252329287 */
        /*03c8*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*03d0*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*03d8*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
        /*03e0*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*03e8*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*03f0*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*03f8*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
                                                                                              /* 0x22b2c27200429237 */
        /*0408*/                   DFMA R18, R12, R14, R16;                                   /* 0x2020000038c49c01 */
        /*0410*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
        /*0418*/                   FFMA R8, RZ, R7, R19;                                      /* 0x302600001ff21c00 */
        /*0420*/                   ST.E.64 [R2+0x2ee18], R4;                                  /* 0x94000bb860211ca5 */
        /*0428*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*0430*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003083dc80 */
        /*0438*/               @P1 BRA CC.NEU, 0x470;                                         /* 0x40000000c00005a7 */
                                                                                              /* 0x2232b2b042b2b047 */
        /*0448*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*0450*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*0458*/                   CAL 0xaf0;                                                 /* 0x5000001a40011c07 */
        /*0460*/                   MOV R18, R8;                                               /* 0x2800000020049de4 */
        /*0468*/                   MOV R19, R9;                                               /* 0x280000002404dde4 */
        /*0470*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0478*/                   LD.E.64 R12, [R2+0x3e818];                                 /* 0x84000fa060231ca5 */
                                                                                              /* 0x22e203f232323047 */
        /*0488*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
        /*0490*/                   SSY 0x598;                                                 /* 0x6000000400001c07 */
        /*0498*/                   LD.E.64 R16, [R2+0x4e218];                                 /* 0x8400138860241ca5 */
        /*04a0*/                   LD.E.64 R4, [R2+0x2ee18];                                  /* 0x84000bb860211ca5 */
        /*04a8*/                   LD.E.64 R8, [R2+0xfa18];                                   /* 0x840003e860221ca5 */
        /*04b0*/                   DADD R14, R18, R12;                                        /* 0x4800000031239c01 */
        /*04b8*/                   LD.E.64 R12, [R2+0x1f418];                                 /* 0x840007d060231ca5 */
                                                                                              /* 0x229280423042e047 */
        /*04c8*/                   ST.E.64 [R2+0x3e818], R14;                                 /* 0x94000fa060239ca5 */
        /*04d0*/                   MUFU.RCP64H R19, R7;                                       /* 0xc80000001874dc00 */
        /*04d8*/                   LD.E.64 R10, [R2+0x3e818];                                 /* 0x84000fa060229ca5 */
        /*04e0*/                   DMUL R16, R16, c[0x0][0x148];                              /* 0x5000400521041c01 */
        /*04e8*/                   MOV32I R18, 0x1;                                           /* 0x1800000004049de2 */
        /*04f0*/                   DFMA R4, R4, c[0x0][0x148], R8;                            /* 0x2010400520411c01 */
        /*04f8*/                   DFMA R20, -R6, R18, c[0x2][0x0];                           /* 0x2024880000651e01 */
                                                                                              /* 0x23f0420292929297 */
        /*0508*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*0510*/                   DFMA R18, R18, R20, R18;                                   /* 0x2024000051249c01 */
        /*0518*/                   DMUL R20, R16, R18;                                        /* 0x5000000049051c01 */
        /*0520*/                   DFMA R8, -R6, R20, R16;                                    /* 0x2020000050621e01 */
        /*0528*/                   DFMA R8, R18, R8, R20;                                     /* 0x2028000021221c01 */
        /*0530*/                   ST.E.64 [R2+0xfa18], R4;                                   /* 0x940003e860211ca5 */
        /*0538*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                /* 0x160f4800211fdc80 */
                                                                                              /* 0x22b042b2c0428207 */
        /*0548*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*0550*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*0558*/                   ST.E.64 [R2+0x1f418], R12;                                 /* 0x940007d060231ca5 */
        /*0560*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*0568*/               @P1 BRA CC.NEU, 0x590;                                         /* 0x40000000800005a7 */
        /*0570*/                   MOV R4, R16;                                               /* 0x2800000040011de4 */
        /*0578*/                   MOV R5, R17;                                               /* 0x2800000044015de4 */
                                                                                              /* 0x23f230423042b2b7 */
        /*0588*/                   CAL 0xaf0;                                                 /* 0x5000001580011c07 */
        /*0590*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0598*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
        /*05a0*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*05a8*/                   LD.E.64 R10, [R2+0x5dc18];                                 /* 0x8400177060229ca5 */
        /*05b0*/                   SSY 0x678;                                                 /* 0x6000000300001c07 */
        /*05b8*/                   LD.E.64 R4, [R2+0x2ee18];                                  /* 0x84000bb860211ca5 */
                                                                                              /* 0x2292929252329287 */
        /*05c8*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*05d0*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*05d8*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
        /*05e0*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*05e8*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*05f0*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*05f8*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
                                                                                              /* 0x22b2c27200429237 */
        /*0608*/                   DFMA R18, R12, R14, R16;                                   /* 0x2020000038c49c01 */
        /*0610*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
        /*0618*/                   FFMA R8, RZ, R7, R19;                                      /* 0x302600001ff21c00 */
        /*0620*/                   ST.E.64 [R2+0x2ee18], R4;                                  /* 0x94000bb860211ca5 */
        /*0628*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*0630*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003083dc80 */
        /*0638*/               @P1 BRA CC.NEU, 0x670;                                         /* 0x40000000c00005a7 */
                                                                                              /* 0x2232b2b042b2b047 */
        /*0648*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*0650*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*0658*/                   CAL 0xaf0;                                                 /* 0x5000001240011c07 */
        /*0660*/                   MOV R18, R8;                                               /* 0x2800000020049de4 */
        /*0668*/                   MOV R19, R9;                                               /* 0x280000002404dde4 */
        /*0670*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0678*/                   LD.E.64 R12, [R2+0x3e818];                                 /* 0x84000fa060231ca5 */
                                                                                              /* 0x22e203f232323047 */
        /*0688*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
        /*0690*/                   SSY 0x798;                                                 /* 0x6000000400001c07 */
        /*0698*/                   LD.E.64 R16, [R2+0x4e218];                                 /* 0x8400138860241ca5 */
        /*06a0*/                   LD.E.64 R4, [R2+0x2ee18];                                  /* 0x84000bb860211ca5 */
        /*06a8*/                   LD.E.64 R8, [R2+0xfa18];                                   /* 0x840003e860221ca5 */
        /*06b0*/                   DADD R14, R18, R12;                                        /* 0x4800000031239c01 */
        /*06b8*/                   LD.E.64 R12, [R2+0x1f418];                                 /* 0x840007d060231ca5 */
                                                                                              /* 0x229280423042e047 */
        /*06c8*/                   ST.E.64 [R2+0x3e818], R14;                                 /* 0x94000fa060239ca5 */
        /*06d0*/                   MUFU.RCP64H R19, R7;                                       /* 0xc80000001874dc00 */
        /*06d8*/                   LD.E.64 R10, [R2+0x3e818];                                 /* 0x84000fa060229ca5 */
        /*06e0*/                   DMUL R16, R16, c[0x0][0x148];                              /* 0x5000400521041c01 */
        /*06e8*/                   MOV32I R18, 0x1;                                           /* 0x1800000004049de2 */
        /*06f0*/                   DFMA R4, R4, c[0x0][0x148], R8;                            /* 0x2010400520411c01 */
        /*06f8*/                   DFMA R20, -R6, R18, c[0x2][0x0];                           /* 0x2024880000651e01 */
                                                                                              /* 0x23f0420292929297 */
        /*0708*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*0710*/                   DFMA R18, R18, R20, R18;                                   /* 0x2024000051249c01 */
        /*0718*/                   DMUL R20, R16, R18;                                        /* 0x5000000049051c01 */
        /*0720*/                   DFMA R8, -R6, R20, R16;                                    /* 0x2020000050621e01 */
        /*0728*/                   DFMA R8, R18, R8, R20;                                     /* 0x2028000021221c01 */
        /*0730*/                   ST.E.64 [R2+0xfa18], R4;                                   /* 0x940003e860211ca5 */
        /*0738*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                /* 0x160f4800211fdc80 */
                                                                                              /* 0x22b042b2c0428207 */
        /*0748*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*0750*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*0758*/                   ST.E.64 [R2+0x1f418], R12;                                 /* 0x940007d060231ca5 */
        /*0760*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*0768*/               @P1 BRA CC.NEU, 0x790;                                         /* 0x40000000800005a7 */
        /*0770*/                   MOV R4, R16;                                               /* 0x2800000040011de4 */
        /*0778*/                   MOV R5, R17;                                               /* 0x2800000044015de4 */
                                                                                              /* 0x23f230423042b2b7 */
        /*0788*/                   CAL 0xaf0;                                                 /* 0x5000000d80011c07 */
        /*0790*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0798*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
        /*07a0*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*07a8*/                   LD.E.64 R10, [R2+0x5dc18];                                 /* 0x8400177060229ca5 */
        /*07b0*/                   SSY 0x878;                                                 /* 0x6000000300001c07 */
        /*07b8*/                   LD.E.64 R4, [R2+0x2ee18];                                  /* 0x84000bb860211ca5 */
                                                                                              /* 0x2292929252329287 */
        /*07c8*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*07d0*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*07d8*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
        /*07e0*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*07e8*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*07f0*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*07f8*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
                                                                                              /* 0x22b2c27200429237 */
        /*0808*/                   DFMA R18, R12, R14, R16;                                   /* 0x2020000038c49c01 */
        /*0810*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
        /*0818*/                   FFMA R8, RZ, R7, R19;                                      /* 0x302600001ff21c00 */
        /*0820*/                   ST.E.64 [R2+0x2ee18], R4;                                  /* 0x94000bb860211ca5 */
        /*0828*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*0830*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003083dc80 */
        /*0838*/               @P1 BRA CC.NEU, 0x870;                                         /* 0x40000000c00005a7 */
                                                                                              /* 0x2232b2b042b2b047 */
        /*0848*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*0850*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*0858*/                   CAL 0xaf0;                                                 /* 0x5000000a40011c07 */
        /*0860*/                   MOV R18, R8;                                               /* 0x2800000020049de4 */
        /*0868*/                   MOV R19, R9;                                               /* 0x280000002404dde4 */
        /*0870*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0878*/                   LD.E.64 R12, [R2+0x3e818];                                 /* 0x84000fa060231ca5 */
                                                                                              /* 0x22e203f232323047 */
        /*0888*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
        /*0890*/                   SSY 0x998;                                                 /* 0x6000000400001c07 */
        /*0898*/                   LD.E.64 R16, [R2+0x4e218];                                 /* 0x8400138860241ca5 */
        /*08a0*/                   LD.E.64 R4, [R2+0x2ee18];                                  /* 0x84000bb860211ca5 */
        /*08a8*/                   LD.E.64 R8, [R2+0xfa18];                                   /* 0x840003e860221ca5 */
        /*08b0*/                   DADD R14, R18, R12;                                        /* 0x4800000031239c01 */
        /*08b8*/                   LD.E.64 R12, [R2+0x1f418];                                 /* 0x840007d060231ca5 */
                                                                                              /* 0x229280423042e047 */
        /*08c8*/                   ST.E.64 [R2+0x3e818], R14;                                 /* 0x94000fa060239ca5 */
        /*08d0*/                   MUFU.RCP64H R19, R7;                                       /* 0xc80000001874dc00 */
        /*08d8*/                   LD.E.64 R10, [R2+0x3e818];                                 /* 0x84000fa060229ca5 */
        /*08e0*/                   DMUL R16, R16, c[0x0][0x148];                              /* 0x5000400521041c01 */
        /*08e8*/                   MOV32I R18, 0x1;                                           /* 0x1800000004049de2 */
        /*08f0*/                   DFMA R4, R4, c[0x0][0x148], R8;                            /* 0x2010400520411c01 */
        /*08f8*/                   DFMA R20, -R6, R18, c[0x2][0x0];                           /* 0x2024880000651e01 */
                                                                                              /* 0x23f0420292929297 */
        /*0908*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*0910*/                   DFMA R18, R18, R20, R18;                                   /* 0x2024000051249c01 */
        /*0918*/                   DMUL R20, R16, R18;                                        /* 0x5000000049051c01 */
        /*0920*/                   DFMA R8, -R6, R20, R16;                                    /* 0x2020000050621e01 */
        /*0928*/                   DFMA R8, R18, R8, R20;                                     /* 0x2028000021221c01 */
        /*0930*/                   ST.E.64 [R2+0xfa18], R4;                                   /* 0x940003e860211ca5 */
        /*0938*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                /* 0x160f4800211fdc80 */
                                                                                              /* 0x22b042b2c0428207 */
        /*0948*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*0950*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*0958*/                   ST.E.64 [R2+0x1f418], R12;                                 /* 0x940007d060231ca5 */
        /*0960*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*0968*/               @P1 BRA CC.NEU, 0x990;                                         /* 0x40000000800005a7 */
        /*0970*/                   MOV R4, R16;                                               /* 0x2800000040011de4 */
        /*0978*/                   MOV R5, R17;                                               /* 0x2800000044015de4 */
                                                                                              /* 0x23f230423042b2b7 */
        /*0988*/                   CAL 0xaf0;                                                 /* 0x5000000580011c07 */
        /*0990*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0998*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
        /*09a0*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*09a8*/                   LD.E.64 R10, [R2+0x5dc18];                                 /* 0x8400177060229ca5 */
        /*09b0*/                   SSY 0xa78;                                                 /* 0x6000000300001c07 */
        /*09b8*/                   LD.E.64 R4, [R2+0x2ee18];                                  /* 0x84000bb860211ca5 */
                                                                                              /* 0x2292929252329287 */
        /*09c8*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*09d0*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*09d8*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
        /*09e0*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*09e8*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*09f0*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*09f8*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
                                                                                              /* 0x22b2c27200429237 */
        /*0a08*/                   DFMA R16, R12, R14, R16;                                   /* 0x2020000038c41c01 */
        /*0a10*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
        /*0a18*/                   FFMA R8, RZ, R7, R17;                                      /* 0x302200001ff21c00 */
        /*0a20*/                   ST.E.64 [R2+0x2ee18], R4;                                  /* 0x94000bb860211ca5 */
        /*0a28*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*0a30*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003083dc80 */
        /*0a38*/               @P1 BRA CC.NEU, 0xa70;                                         /* 0x40000000c00005a7 */
                                                                                              /* 0x2232b2b042b2b047 */
        /*0a48*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*0a50*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*0a58*/                   CAL 0xaf0;                                                 /* 0x5000000240011c07 */
        /*0a60*/                   MOV R16, R8;                                               /* 0x2800000020041de4 */
        /*0a68*/                   MOV R17, R9;                                               /* 0x2800000024045de4 */
        /*0a70*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0a78*/                   LD.E.64 R4, [R2+0x3e818];                                  /* 0x84000fa060211ca5 */
                                                                                              /* 0x22e2e04293f23237 */
        /*0a88*/                   LD.E.64 R6, [R2+0x2ee18];                                  /* 0x84000bb860219ca5 */
        /*0a90*/                   LD.E.64 R8, [R2+0xfa18];                                   /* 0x840003e860221ca5 */
        /*0a98*/                   LD.E.64 R12, [R2+0x1f418];                                 /* 0x840007d060231ca5 */
        /*0aa0*/                   DADD R4, R16, R4;                                          /* 0x4800000011011c01 */
        /*0aa8*/                   ST.E.64 [R2+0x3e818], R4;                                  /* 0x94000fa060211ca5 */
        /*0ab0*/                   DFMA R6, R6, c[0x0][0x148], R8;                            /* 0x2010400520619c01 */
        /*0ab8*/                   LD.E.64 R10, [R2+0x3e818];                                 /* 0x84000fa060229ca5 */
                                                                                              /* 0x228042e2e04293f7 */
        /*0ac8*/                   ST.E.64 [R2+0xfa18], R6;                                   /* 0x940003e860219ca5 */
        /*0ad0*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*0ad8*/                   ST.E.64 [R2+0x1f418], R12;                                 /* 0x940007d060231ca5 */
        /*0ae0*/               @P0 BRA 0xe0;                                                  /* 0x4003ffd7e00001e7 */
        /*0ae8*/                   EXIT;                                                      /* 0x8000000000001de7 */
        /*0af0*/                   LOP32I.AND R8, R7, 0x40000000;                             /* 0x3900000000721c02 */
        /*0af8*/                   MOV32I R12, 0x1ff00000;                                    /* 0x187fc00000031de2 */
                                                                                              /* 0x22028042b0420047 */
        /*0b08*/                   ISETP.LT.U32.AND P1, PT, R8, c[0x2][0x10], PT;             /* 0x188e48004083dc03 */
        /*0b10*/                   LOP32I.AND R13, R5, 0x7f800000;                            /* 0x39fe000000535c02 */
        /*0b18*/                   MOV R8, RZ;                                                /* 0x28000000fc021de4 */
        /*0b20*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0b28*/                   SEL R9, R12, c[0x2][0x14], !P1;                            /* 0x2012480050c25c04 */
        /*0b30*/                   PBK 0xde8;                                                 /* 0x6800000ac0001c07 */
        /*0b38*/                   DMUL R10, R6, R8;                                          /* 0x5000000020629c01 */
                                                                                              /* 0x225232928042c047 */
        /*0b48*/                   ISETP.LT.U32.AND P1, PT, R13, c[0x2][0x18], PT;            /* 0x188e480060d3dc03 */
        /*0b50*/                   MOV R16, RZ;                                               /* 0x28000000fc041de4 */
        /*0b58*/                   MUFU.RCP64H R15, R11;                                      /* 0xc800000018b3dc00 */
        /*0b60*/                   SEL R17, R12, c[0x2][0x14], !P1;                           /* 0x2012480050c45c04 */
        /*0b68*/                   DFMA R18, -R10, R14, c[0x2][0x0];                          /* 0x201c880000a49e01 */
        /*0b70*/                   DFMA R18, R18, R18, R18;                                   /* 0x2024000049249c01 */
        /*0b78*/                   DMUL R16, R4, R16;                                         /* 0x5000000040441c01 */
                                                                                              /* 0x2202b2d292929297 */
        /*0b88*/                   DFMA R14, R14, R18, R14;                                   /* 0x201c000048e39c01 */
        /*0b90*/                   DMUL R20, R16, R14;                                        /* 0x5000000039051c01 */
        /*0b98*/                   DFMA R10, -R10, R20, R16;                                  /* 0x2020000050a29e01 */
        /*0ba0*/                   DFMA R14, R14, R10, R20;                                   /* 0x2028000028e39c01 */
        /*0ba8*/                   DSETP.LEU.AND P1, PT, |R14|, RZ, PT;                       /* 0x1d8e0000fce3dc81 */
        /*0bb0*/               @P1 BRA 0xd68;                                                 /* 0x40000006c00005e7 */
        /*0bb8*/                   ISETP.GT.U32.AND P1, PT, R13, c[0x2][0x1c], PT;            /* 0x1a0e480070d3dc03 */
                                                                                              /* 0x22523252327202b7 */
        /*0bc8*/                   MOV R10, RZ;                                               /* 0x28000000fc029de4 */
        /*0bd0*/                   SEL R11, R12, c[0x2][0x14], !P1;                           /* 0x2012480050c2dc04 */
        /*0bd8*/                   DMUL R12, R8, R14;                                         /* 0x5000000038831c01 */
        /*0be0*/                   DMUL R14, R14, R10;                                        /* 0x5000000028e39c01 */
        /*0be8*/                   DMUL R10, R10, R12;                                        /* 0x5000000030a29c01 */
        /*0bf0*/                   DMUL R14, R8, R14;                                         /* 0x5000000038839c01 */
        /*0bf8*/                   DFMA R8, R6, R10, -R4;                                     /* 0x2008000028621d01 */
                                                                                              /* 0x2202b2c27202d297 */
        /*0c08*/                   DFMA R12, R6, R14, -R4;                                    /* 0x2008000038631d01 */
        /*0c10*/                   DSETP.GT.AND P1, PT, |R8|, |R12|, PT;                      /* 0x1a0e00003083dcc1 */
        /*0c18*/                   SEL R9, R15, R11, P1;                                      /* 0x200200002cf25c04 */
        /*0c20*/                   SEL R8, R14, R10, P1;                                      /* 0x2002000028e21c04 */
        /*0c28*/                   FSETP.GTU.AND P1, PT, |R9|, 1.469367938527859385e-39, PT;  /* 0x260ec0040093dc80 */
        /*0c30*/               @P1 BRK;                                                       /* 0xa8000000000005e7 */
        /*0c38*/                   FSETP.LT.AND P1, PT, |R5|, 1.5046327690525280102e-36, PT;  /* 0x208ec1000053dc80 */
                                                                                              /* 0x2232804270423047 */
        /*0c48*/                   MOV32I R11, 0x3ff00000;                                    /* 0x18ffc0000002dde2 */
        /*0c50*/                   LOP32I.AND R14, R8, 0xfffffffe;                            /* 0x3bfffffff8839c02 */
        /*0c58*/                   MOV R10, RZ;                                               /* 0x28000000fc029de4 */
        /*0c60*/                   LOP32I.OR R15, R8, 0x1;                                    /* 0x380000000483dc42 */
        /*0c68*/                   SEL R11, R11, c[0x2][0x20], !P1;                           /* 0x2012480080b2dc04 */
        /*0c70*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0c78*/                   DMUL R6, R6, R10;                                          /* 0x5000000028619c01 */
                                                                                              /* 0x2282d29232720207 */
        /*0c88*/                   DMUL R4, R4, R10;                                          /* 0x5000000028411c01 */
        /*0c90*/                   MOV R10, R15;                                              /* 0x280000003c029de4 */
        /*0c98*/                   MOV R11, R9;                                               /* 0x280000002402dde4 */
        /*0ca0*/                   DFMA R12, R8, R6, -R4;                                     /* 0x2008000018831d01 */
        /*0ca8*/                   DFMA R10, R10, R6, -R4;                                    /* 0x2008000018a29d01 */
        /*0cb0*/                   DSETP.GT.AND P1, PT, |R12|, |R10|, PT;                     /* 0x1a0e000028c3dcc1 */
        /*0cb8*/                   SEL R15, R15, R14, P1;                                     /* 0x2002000038f3dc04 */
                                                                                              /* 0x220042c20042c047 */
        /*0cc8*/                   IADD32I R12.CC, R15, 0x1;                                  /* 0x0c00000004f31c02 */
        /*0cd0*/                   LOP32I.AND R8, R15, 0x1;                                   /* 0x3800000004f21c02 */
        /*0cd8*/                   ISETP.EQ.U32.AND P1, PT, R8, 0x1, PT;                      /* 0x190ec0000483dc03 */
        /*0ce0*/                   IADD.X R13, RZ, R9;                                        /* 0x4800000027f35c43 */
        /*0ce8*/                   IADD32I R10.CC, R15, -0x1;                                 /* 0x0ffffffffcf29c02 */
        /*0cf0*/                   SEL R12, R15, R12, !P1;                                    /* 0x2012000030f31c04 */
        /*0cf8*/                   IADD32I.X R8, R9, -0x1;                                    /* 0x0bfffffffc921c42 */
                                                                                              /* 0x2292328202026207 */
        /*0d08*/                   SEL R13, R9, R13, !P1;                                     /* 0x2012000034935c04 */
        /*0d10*/                   SEL R10, R10, R15, !P1;                                    /* 0x201200003ca29c04 */
        /*0d18*/                   SEL R11, R8, R9, !P1;                                      /* 0x201200002482dc04 */
        /*0d20*/                   MOV R8, R12;                                               /* 0x2800000030021de4 */
        /*0d28*/                   MOV R9, R13;                                               /* 0x2800000034025de4 */
        /*0d30*/                   DFMA R8, R6, R8, -R4;                                      /* 0x2008000020621d01 */
        /*0d38*/                   DFMA R4, R6, R10, -R4;                                     /* 0x2008000028611d01 */
                                                                                              /* 0x2202b2d2e04202d7 */
        /*0d48*/                   DSETP.GT.AND P1, PT, |R8|, |R4|, PT;                       /* 0x1a0e00001083dcc1 */
        /*0d50*/                   SEL R8, R10, R12, P1;                                      /* 0x2002000030a21c04 */
        /*0d58*/                   SEL R9, R11, R13, P1;                                      /* 0x2002000034b25c04 */
        /*0d60*/                   BRK;                                                       /* 0xa800000000001de7 */
        /*0d68*/                   DSETP.EQ.AND P1, PT, R14, RZ, PT;                          /* 0x190e0000fce3dc01 */
        /*0d70*/               @P1 BRA 0xdd8;                                                 /* 0x40000001800005e7 */
        /*0d78*/                   MUFU.RCP64H R9, R7;                                        /* 0xc800000018725c00 */
                                                                                              /* 0x2282027202d2d287 */
        /*0d88*/                   MOV R8, RZ;                                                /* 0x28000000fc021de4 */
        /*0d90*/                   DSETP.GT.AND P2, PT, |R8|, RZ, PT;                         /* 0x1a0e0000fc85dc81 */
        /*0d98*/              @!P2 DSETP.NEU.AND P1, PT, |R6|, +INF , PT;                     /* 0x1e8edffc0063e881 */
        /*0da0*/              @!P2 SEL R6, R6, R8, P1;                                        /* 0x200200002061a804 */
        /*0da8*/              @!P2 SEL R7, R7, R9, P1;                                        /* 0x200200002471e804 */
        /*0db0*/              @!P2 MOV R8, R6;                                                /* 0x28000000180229e4 */
        /*0db8*/              @!P2 MOV R9, R7;                                                /* 0x280000001c0269e4 */
                                                                                              /* 0x200002e2b202e207 */
        /*0dc8*/                   DMUL R8, R4, R8;                                           /* 0x5000000020421c01 */
        /*0dd0*/                   BRK;                                                       /* 0xa800000000001de7 */
        /*0dd8*/                   DMUL R8, R4, R6;                                           /* 0x5000000018421c01 */
        /*0de0*/                   BRK;                                                       /* 0xa800000000001de7 */
        /*0de8*/                   RET;                                                       /* 0x9000000000001de7 */
        /*0df0*/                   BRA 0xdf0;                                                 /* 0x4003ffffe0001de7 */
        /*0df8*/                   NOP;                                                       /* 0x4000000000001de4 */
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
                                                                                               /* 0x22c2024230423047 */
        /*0088*/                   IADD32I RZ.CC, R0, -0x20;                                   /* 0x0fffffff800fdc02 */
        /*0090*/                   SHL R5, R5, 0x3;                                            /* 0x6000c0000c515c03 */
        /*0098*/                   SHL R0, R4, -0x1d;                                          /* 0x6000ffff8c401c03 */
        /*00a0*/                   SSY 0x1a0;                                                  /* 0x60000003e0001c07 */
        /*00a8*/                   SHR.U32 R2, R4, 0x1d;                                       /* 0x5800c00074409c03 */
        /*00b0*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*00b8*/                   IMAD.U32.U32 R4.CC, R4, 0x8, R7;                            /* 0x200fc00020411c03 */
                                                                                               /* 0x2283f23042828047 */
        /*00c8*/               @P0 IADD R0, R5, R2;                                            /* 0x4800000008500003 */
        /*00d0*/                   MOV R2, R4;                                                 /* 0x2800000010009de4 */
        /*00d8*/                   IADD.X R3, R0, R6;                                          /* 0x480000001800dc43 */
        /*00e0*/                   LD.E.64 R8, [R2+0x18];                                      /* 0x8400000060221ca5 */
        /*00e8*/                   MOV32I R6, 0x1;                                             /* 0x1800000004019de2 */
        /*00f0*/                   LD.E.64 R4, [R2+0x4e218];                                   /* 0x8400138860211ca5 */
        /*00f8*/                   MUFU.RCP64H R7, R9;                                         /* 0xc80000001891dc00 */
                                                                                               /* 0x2202929292523297 */
        /*0108*/                   DFMA R10, -R8, R6, c[0x2][0x0];                             /* 0x200c880000829e01 */
        /*0110*/                   DFMA R10, R10, R10, R10;                                    /* 0x2014000028a29c01 */
        /*0118*/                   DMUL R14, R4, c[0x0][0x148];                                /* 0x5000400520439c01 */
        /*0120*/                   DFMA R10, R6, R10, R6;                                      /* 0x200c000028629c01 */
        /*0128*/                   DMUL R12, R14, R10;                                         /* 0x5000000028e31c01 */
        /*0130*/                   DFMA R6, -R8, R12, R14;                                     /* 0x201c000030819e01 */
        /*0138*/                   DFMA R12, R10, R6, R12;                                     /* 0x2018000018a31c01 */
                                                                                               /* 0x22b2b042b2c28287 */
        /*0148*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                 /* 0x160f480020ffdc80 */
        /*0150*/                   FFMA R0, RZ, R9, R13;                                       /* 0x301a000027f01c00 */
        /*0158*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0xc], PT;                 /* 0x220e48003001dc80 */
        /*0160*/               @P0 BRA CC.NEU, 0x198;                                          /* 0x40000000c00001a7 */
        /*0168*/                   MOV R10, R8;                                                /* 0x2800000020029de4 */
        /*0170*/                   MOV R11, R9;                                                /* 0x280000002402dde4 */
        /*0178*/                   CAL 0x358;                                                  /* 0x5000000760011c07 */
                                                                                               /* 0x2230423042b2b047 */
        /*0188*/                   MOV R12, R22;                                               /* 0x2800000058031de4 */
        /*0190*/                   MOV R13, R23;                                               /* 0x280000005c035de4 */
        /*0198*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*01a0*/                   LD.E.64 R10, [R2+0x18];                                     /* 0x8400000060229ca5 */
        /*01a8*/                   MOV32I R20, 0x1;                                            /* 0x1800000004051de2 */
        /*01b0*/                   LD.E.64 R16, [R2+0x5dc18];                                  /* 0x8400177060241ca5 */
        /*01b8*/                   SSY 0x290;                                                  /* 0x6000000340001c07 */
                                                                                               /* 0x22329283f2323237 */
        /*01c8*/                   LD.E.64 R14, [R2+0x2ee18];                                  /* 0x84000bb860239ca5 */
        /*01d0*/                   LD.E.64 R4, [R2+0x3e818];                                   /* 0x84000fa060211ca5 */
        /*01d8*/                   LD.E.64 R6, [R2+0xfa18];                                    /* 0x840003e860219ca5 */
        /*01e0*/                   LD.E.64 R8, [R2+0x1f418];                                   /* 0x840007d060221ca5 */
        /*01e8*/                   MUFU.RCP64H R21, R11;                                       /* 0xc800000018b55c00 */
        /*01f0*/                   DFMA R18, -R10, R20, c[0x2][0x0];                           /* 0x2028880000a49e01 */
        /*01f8*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
                                                                                               /* 0x2232523292929257 */
        /*0208*/                   DMUL R16, R16, c[0x0][0x148];                               /* 0x5000400521041c01 */
        /*0210*/                   DFMA R18, R20, R18, R20;                                    /* 0x2028000049449c01 */
        /*0218*/                   DMUL R20, R16, R18;                                         /* 0x5000000049051c01 */
        /*0220*/                   DFMA R22, -R10, R20, R16;                                   /* 0x2020000050a59e01 */
        /*0228*/                   DFMA R22, R18, R22, R20;                                    /* 0x2028000059259c01 */
        /*0230*/                   DADD R12, R12, R14;                                         /* 0x4800000038c31c01 */
        /*0238*/                   FFMA R0, RZ, R11, R23;                                      /* 0x302e00002ff01c00 */
                                                                                               /* 0x22b2b042b2c24047 */
        /*0248*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                 /* 0x160f4800211fdc80 */
        /*0250*/                   ST.E.64 [R2+0x2ee18], R12;                                  /* 0x94000bb860231ca5 */
        /*0258*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0xc], PT;                 /* 0x220e48003001dc80 */
        /*0260*/               @P0 BRA CC.NEU, 0x280;                                          /* 0x40000000600001a7 */
        /*0268*/                   MOV R14, R16;                                               /* 0x2800000040039de4 */
        /*0270*/                   MOV R15, R17;                                               /* 0x280000004403dde4 */
        /*0278*/                   CAL 0x358;                                                  /* 0x5000000360011c07 */
                                                                                               /* 0x23f232e2e04202b7 */
        /*0288*/                   DADD.S R4, R4, R22;                                         /* 0x4800000058411c11 */
        /*0290*/                   DFMA R6, R12, c[0x0][0x148], R6;                            /* 0x200c400520c19c01 */
        /*0298*/                   ST.E.64 [R2+0x3e818], R4;                                   /* 0x94000fa060211ca5 */
        /*02a0*/                   SSY 0x318;                                                  /* 0x60000001c0001c07 */
        /*02a8*/                   ST.E.64 [R2+0xfa18], R6;                                    /* 0x940003e860219ca5 */
        /*02b0*/                   LD.E.64 R12, [R2+0xfa18];                                   /* 0x840003e860231ca5 */
        /*02b8*/                   LD.E.64 R10, [R2+0x3e818];                                  /* 0x84000fa060229ca5 */
                                                                                               /* 0x23f232b2d0429237 */
        /*02c8*/                   DSETP.GT.AND P0, PT, R12, 1, PT;                            /* 0x1a0ecffc00c1dc01 */
        /*02d0*/                   DFMA R8, R10, c[0x0][0x148], R8;                            /* 0x2010400520a21c01 */
        /*02d8*/                   ST.E.64 [R2+0x1f418], R8;                                   /* 0x940007d060221ca5 */
        /*02e0*/                   DSETP.LT.OR P0, PT, R12, -1, P0;                            /* 0x18a0effc00c1dc01 */
        /*02e8*/              @!P0 NOP.S;                                                      /* 0x40000000000021f4 */
        /*02f0*/                   LD.E.64 R4, [R2+0x2ee18];                                   /* 0x84000bb860211ca5 */
        /*02f8*/                   LD.E.64 R8, [R2+0x1f418];                                   /* 0x840007d060221ca5 */
                                                                                               /* 0x2283f2e2d2d2b287 */
        /*0308*/                   F2F.F64.F64 R4, -R4;                                        /* 0x1000000011b11d04 */
        /*0310*/                   ST.E.64.S [R2+0x2ee18], R4;                                 /* 0x94000bb860211cb5 */
        /*0318*/                   DSETP.GT.AND P0, PT, R8, 1, PT;                             /* 0x1a0ecffc0081dc01 */
        /*0320*/                   DSETP.LT.OR P0, PT, R8, -1, P0;                             /* 0x18a0effc0081dc01 */
        /*0328*/              @!P0 EXIT;                                                       /* 0x80000000000021e7 */
        /*0330*/                   LD.E.64 R4, [R2+0x3e818];                                   /* 0x84000fa060211ca5 */
        /*0338*/                   F2F.F64.F64 R4, -R4;                                        /* 0x1000000011b11d04 */
                                                                                               /* 0x22b200428042e047 */
        /*0348*/                   ST.E.64 [R2+0x3e818], R4;                                   /* 0x94000fa060211ca5 */
        /*0350*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*0358*/                   LOP32I.AND R0, R11, 0x40000000;                             /* 0x3900000000b01c02 */
        /*0360*/                   LOP32I.AND R26, R15, 0x7f800000;                            /* 0x39fe000000f69c02 */
        /*0368*/                   ISETP.LT.U32.AND P0, PT, R0, c[0x2][0x10], PT;              /* 0x188e48004001dc03 */
        /*0370*/                   MOV32I R0, 0x1ff00000;                                      /* 0x187fc00000001de2 */
        /*0378*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
                                                                                               /* 0x228042c042028047 */
        /*0388*/                   SEL R17, R0, c[0x2][0x14], !P0;                             /* 0x2010480050045c04 */
        /*0390*/                   MOV32I R20, 0x1;                                            /* 0x1800000004051de2 */
        /*0398*/                   DMUL R18, R10, R16;                                         /* 0x5000000040a49c01 */
        /*03a0*/                   ISETP.LT.U32.AND P0, PT, R26, c[0x2][0x18], PT;             /* 0x188e480061a1dc03 */
        /*03a8*/                   MOV R24, RZ;                                                /* 0x28000000fc061de4 */
        /*03b0*/                   MUFU.RCP64H R21, R19;                                       /* 0xc800000019355c00 */
        /*03b8*/                   SEL R25, R0, c[0x2][0x14], !P0;                             /* 0x2010480050065c04 */
                                                                                               /* 0x2292929292523297 */
        /*03c8*/                   DFMA R22, -R18, R20, c[0x2][0x0];                           /* 0x2028880001259e01 */
        /*03d0*/                   DFMA R30, R22, R22, R22;                                    /* 0x202c000059679c01 */
        /*03d8*/                   DMUL R22, R14, R24;                                         /* 0x5000000060e59c01 */
        /*03e0*/                   DFMA R20, R20, R30, R20;                                    /* 0x2028000079451c01 */
        /*03e8*/                   DMUL R24, R22, R20;                                         /* 0x5000000051661c01 */
        /*03f0*/                   DFMA R18, -R18, R24, R22;                                   /* 0x202c000061249e01 */
        /*03f8*/                   DFMA R22, R20, R18, R24;                                    /* 0x2030000049459c01 */
                                                                                               /* 0x223282823042b2d7 */
        /*0408*/                   DSETP.LEU.AND P0, PT, |R22|, RZ, PT;                        /* 0x1d8e0000fd61dc81 */
        /*0410*/               @P0 BRA 0x5c0;                                                  /* 0x40000006a00001e7 */
        /*0418*/                   ISETP.GT.U32.AND P0, PT, R26, c[0x2][0x1c], PT;             /* 0x1a0e480071a1dc03 */
        /*0420*/                   MOV R18, RZ;                                                /* 0x28000000fc049de4 */
        /*0428*/                   DMUL R20, R16, R22;                                         /* 0x5000000059051c01 */
        /*0430*/                   SEL R19, R0, c[0x2][0x14], !P0;                             /* 0x201048005004dc04 */
        /*0438*/                   DMUL R22, R22, R18;                                         /* 0x5000000049659c01 */
                                                                                               /* 0x227202d292523257 */
        /*0448*/                   DMUL R20, R18, R20;                                         /* 0x5000000051251c01 */
        /*0450*/                   DMUL R16, R16, R22;                                         /* 0x5000000059041c01 */
        /*0458*/                   DFMA R18, R10, R20, -R14;                                   /* 0x201c000050a49d01 */
        /*0460*/                   DFMA R22, R10, R16, -R14;                                   /* 0x201c000040a59d01 */
        /*0468*/                   DSETP.GT.AND P0, PT, |R18|, |R22|, PT;                      /* 0x1a0e00005921dcc1 */
        /*0470*/                   SEL R17, R17, R21, P0;                                      /* 0x2000000055145c04 */
        /*0478*/                   SEL R16, R16, R20, P0;                                      /* 0x2000000051041c04 */
                                                                                               /* 0x22a200420042b2c7 */
        /*0488*/                   FSETP.GTU.AND P0, PT, |R17|, 1.469367938527859385e-39, PT;  /* 0x260ec0040111dc80 */
        /*0490*/               @P0 BRA 0x638;                                                  /* 0x40000006800001e7 */
        /*0498*/                   FSETP.LT.AND P0, PT, |R15|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000f1dc80 */
        /*04a0*/                   LOP32I.AND R0, R16, 0xfffffffe;                             /* 0x3bfffffff9001c02 */
        /*04a8*/                   MOV32I R21, 0x3ff00000;                                     /* 0x18ffc00000055de2 */
        /*04b0*/                   MOV R20, RZ;                                                /* 0x28000000fc051de4 */
        /*04b8*/                   LOP32I.OR R22, R16, 0x1;                                    /* 0x3800000005059c42 */
                                                                                               /* 0x2232804202328047 */
        /*04c8*/                   SEL R21, R21, c[0x2][0x20], !P0;                            /* 0x2010480081555c04 */
        /*04d0*/                   MOV R16, R0;                                                /* 0x2800000000041de4 */
        /*04d8*/                   DMUL R10, R10, R20;                                         /* 0x5000000050a29c01 */
        /*04e0*/                   DMUL R14, R14, R20;                                         /* 0x5000000050e39c01 */
        /*04e8*/                   MOV R20, R22;                                               /* 0x2800000058051de4 */
        /*04f0*/                   MOV R21, R17;                                               /* 0x2800000044055de4 */
        /*04f8*/                   DFMA R18, R16, R10, -R14;                                   /* 0x201c000029049d01 */
                                                                                               /* 0x220042c04282d297 */
        /*0508*/                   DFMA R20, R20, R10, -R14;                                   /* 0x201c000029451d01 */
        /*0510*/                   DSETP.GT.AND P0, PT, |R18|, |R20|, PT;                      /* 0x1a0e00005121dcc1 */
        /*0518*/                   SEL R23, R22, R0, P0;                                       /* 0x200000000165dc04 */
        /*0520*/                   IADD32I R0.CC, R23, 0x1;                                    /* 0x0c00000005701c02 */
        /*0528*/                   LOP32I.AND R18, R23, 0x1;                                   /* 0x3800000005749c02 */
        /*0530*/                   ISETP.EQ.U32.AND P0, PT, R18, 0x1, PT;                      /* 0x190ec0000521dc03 */
        /*0538*/                   IADD.X R16, RZ, R17;                                        /* 0x4800000047f41c43 */
                                                                                               /* 0x22020262020042c7 */
        /*0548*/                   IADD32I R20.CC, R23, -0x1;                                  /* 0x0ffffffffd751c02 */
        /*0550*/                   SEL R0, R23, R0, !P0;                                       /* 0x2010000001701c04 */
        /*0558*/                   IADD32I.X R21, R17, -0x1;                                   /* 0x0bfffffffd155c42 */
        /*0560*/                   SEL R19, R17, R16, !P0;                                     /* 0x201000004114dc04 */
        /*0568*/                   SEL R20, R20, R23, !P0;                                     /* 0x201000005d451c04 */
        /*0570*/                   SEL R21, R21, R17, !P0;                                     /* 0x2010000045555c04 */
        /*0578*/                   MOV R16, R0;                                                /* 0x2800000000041de4 */
                                                                                               /* 0x22e04202d2923287 */
        /*0588*/                   MOV R17, R19;                                               /* 0x280000004c045de4 */
        /*0590*/                   DFMA R16, R10, R16, -R14;                                   /* 0x201c000040a41d01 */
        /*0598*/                   DFMA R10, R10, R20, -R14;                                   /* 0x201c000050a29d01 */
        /*05a0*/                   DSETP.GT.AND P0, PT, |R16|, |R10|, PT;                      /* 0x1a0e00002901dcc1 */
        /*05a8*/                   SEL R16, R20, R0, P0;                                       /* 0x2000000001441c04 */
        /*05b0*/                   SEL R17, R21, R19, P0;                                      /* 0x200000004d545c04 */
        /*05b8*/                   BRA 0x638;                                                  /* 0x40000001e0001de7 */
                                                                                               /* 0x2202d2d28042b2d7 */
        /*05c8*/                   DSETP.EQ.AND P0, PT, R22, RZ, PT;                           /* 0x190e0000fd61dc01 */
        /*05d0*/               @P0 BRA 0x630;                                                  /* 0x40000001600001e7 */
        /*05d8*/                   MUFU.RCP64H R17, R11;                                       /* 0xc800000018b45c00 */
        /*05e0*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*05e8*/                   DSETP.GT.AND P1, PT, |R16|, RZ, PT;                         /* 0x1a0e0000fd03dc81 */
        /*05f0*/              @!P1 DSETP.NEU.AND P0, PT, |R10|, +INF , PT;                     /* 0x1e8edffc00a1e481 */
        /*05f8*/              @!P1 SEL R0, R10, R16, P0;                                       /* 0x2000000040a02404 */
                                                                                               /* 0x2202b2e202820277 */
        /*0608*/              @!P1 SEL R11, R11, R17, P0;                                      /* 0x2000000044b2e404 */
        /*0610*/              @!P1 MOV R16, R0;                                                /* 0x28000000000425e4 */
        /*0618*/              @!P1 MOV R17, R11;                                               /* 0x280000002c0465e4 */
        /*0620*/                   DMUL R16, R14, R16;                                         /* 0x5000000040e41c01 */
        /*0628*/                   BRA 0x638;                                                  /* 0x4000000020001de7 */
        /*0630*/                   DMUL R16, R14, R10;                                         /* 0x5000000028e41c01 */
        /*0638*/                   MOV R22, R16;                                               /* 0x2800000040059de4 */
                                                                                               /* 0x200000000002e047 */
        /*0648*/                   MOV R23, R17;                                               /* 0x280000004405dde4 */
        /*0650*/                   RET;                                                        /* 0x9000000000001de7 */
        /*0658*/                   BRA 0x658;                                                  /* 0x4003ffffe0001de7 */
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
        /*0018*/                   S2R R37, SR_TID.X;                                          /* 0x2c00000084095c04 */
        /*0020*/                   MOV32I R5, 0x0;                                             /* 0x1800000000015de2 */
        /*0028*/                   S2R R0, SR_CTAID.X;                                         /* 0x2c00000094001c04 */
        /*0030*/                   LD.E.64 R2, [R4];                                           /* 0x8400000000409ca5 */
        /*0038*/                   IMAD R37, R0, c[0x0][0x28], R37;                            /* 0x204a4000a0095ca3 */
                                                                                               /* 0x22028042e2c2c3f7 */
        /*0048*/                   SHR R36, R37, 0x1f;                                         /* 0x5800c0007e591c23 */
        /*0050*/                   ISUB RZ.CC, R37, R2;                                        /* 0x480100000a5fdd03 */
        /*0058*/                   ISETP.GE.U32.X.AND P0, PT, R36, R3, PT;                     /* 0x1b0e00000e41dc43 */
        /*0060*/               @P0 EXIT;                                                       /* 0x80000000000001e7 */
        /*0068*/                   MOV32I R0, 0x3;                                             /* 0x180000000c001de2 */
        /*0070*/                   SHL R2, R36, 0x3;                                           /* 0x6000c0000e409c03 */
        /*0078*/                   IADD32I RZ.CC, R0, -0x20;                                   /* 0x0fffffff800fdc02 */
                                                                                               /* 0x22b2004270423047 */
        /*0088*/                   SHL R0, R37, -0x1d;                                         /* 0x6000ffff8e501c03 */
        /*0090*/                   MOV32I R43, 0x1f40;                                         /* 0x1800007d000adde2 */
        /*0098*/                   SHR.U32 R3, R37, 0x1d;                                      /* 0x5800c0007650dc03 */
        /*00a0*/                   MOV R44, RZ;                                                /* 0x28000000fc0b1de4 */
        /*00a8*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*00b0*/                   MOV32I R8, 0x5;                                             /* 0x1800000014021de2 */
        /*00b8*/                   IMAD.U32.U32 R9.CC, R37, 0x8, R4;                           /* 0x2009c00022525c03 */
                                                                                               /* 0x2202020200428047 */
        /*00c8*/               @P0 IADD R0, R2, R3;                                            /* 0x480000000c200003 */
        /*00d0*/                   MOV R38, RZ;                                                /* 0x28000000fc099de4 */
        /*00d8*/                   IADD.X R3, R0, R5;                                          /* 0x480000001400dc43 */
        /*00e0*/                   MOV R2, R9;                                                 /* 0x2800000024009de4 */
        /*00e8*/                   IADD32I R33.CC, R37, 0x1;                                   /* 0x0c00000006585c02 */
        /*00f0*/                   MOV32I R7, 0x4;                                             /* 0x180000001001dde2 */
        /*00f8*/                   MOV R39, RZ;                                                /* 0x28000000fc09dde4 */
                                                                                               /* 0x22b2504220420047 */
        /*0108*/                   MOV32I R6, 0x3;                                             /* 0x180000000c019de2 */
        /*0110*/                   MOV R40, RZ;                                                /* 0x28000000fc0a1de4 */
        /*0118*/                   MOV32I R0, 0x2;                                             /* 0x1800000008001de2 */
        /*0120*/                   MOV R41, RZ;                                                /* 0x28000000fc0a5de4 */
        /*0128*/                   MOV32I R32, 0x1;                                            /* 0x1800000004081de2 */
        /*0130*/                   MOV R42, RZ;                                                /* 0x28000000fc0a9de4 */
        /*0138*/                   IADD.X R36, RZ, R36;                                        /* 0x4800000093f91c43 */
                                                                                               /* 0x2202c042c202c047 */
        /*0148*/                   ISUB RZ.CC, R32, R33;                                       /* 0x48010000860fdd03 */
        /*0150*/                   SSY 0x590;                                                  /* 0x60000010e0001c07 */
        /*0158*/                   ISETP.EQ.X.AND P1, PT, R42, R36, PT;                        /* 0x190e000092a3dc63 */
        /*0160*/                   IADD32I R43.CC, R43, -0x5;                                  /* 0x0fffffffeebadc02 */
        /*0168*/                   IADD32I.X R44, R44, -0x1;                                   /* 0x0bfffffffecb1c42 */
        /*0170*/                   IADD RZ.CC, -RZ, R43;                                       /* 0x48010000afffde03 */
        /*0178*/                   ISETP.NE.X.AND P0, PT, R44, RZ, PT;                         /* 0x1a8e0000fec1dc63 */
                                                                                               /* 0x22b04200428042b7 */
        /*0188*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0190*/                   MOV32I R9, 0x3;                                             /* 0x180000000c025de2 */
        /*0198*/                   SHL R12, R32, -0x1d;                                        /* 0x6000ffff8e031c03 */
        /*01a0*/                   IADD32I RZ.CC, R9, -0x20;                                   /* 0x0fffffff809fdc02 */
        /*01a8*/                   SHL R11, R42, 0x3;                                          /* 0x6000c0000ea2dc03 */
        /*01b0*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
        /*01b8*/                   SSY 0x338;                                                  /* 0x60000005e0001c07 */
                                                                                               /* 0x2202b0420042c237 */
        /*01c8*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
        /*01d0*/                   ISCADD R9.CC, R37, 0x8, 0x3;                                /* 0x4001c00022525c63 */
        /*01d8*/                   IMAD.HI.X R10, R37, 0x8, RZ;                                /* 0x20fec00022529ce3 */
        /*01e0*/                   IADD R16.CC, R9, R18;                                       /* 0x4801000048941c03 */
        /*01e8*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*01f0*/                   SHR.U32 R13, R32, 0x1d;                                     /* 0x5800c00076035c03 */
        /*01f8*/                   IADD.X R17, R10, R9;                                        /* 0x4800000024a45c43 */
                                                                                               /* 0x22a232328042c047 */
        /*0208*/                   IMAD.U32.U32 R20.CC, R32, 0x8, R18;                         /* 0x2025c00022051c03 */
        /*0210*/               @P1 IADD R12, R11, R13;                                         /* 0x4800000034b30403 */
        /*0218*/                   IADD.X R21, R12, R9;                                        /* 0x4800000024c55c43 */
        /*0220*/                   LD.E.64 R24, [R16+0x1f410];                                 /* 0x840007d041061ca5 */
        /*0228*/                   LD.E.64 R10, [R20+0x1f410];                                 /* 0x840007d041429ca5 */
        /*0230*/                   LD.E.64 R26, [R16+0xfa10];                                  /* 0x840003e841069ca5 */
        /*0238*/                   LD.E.64 R12, [R20+0xfa10];                                  /* 0x840003e841431ca5 */
                                                                                               /* 0x220292923273f047 */
        /*0248*/                   MOV32I R16, 0x0;                                            /* 0x1800000000041de2 */
        /*0250*/                   MOV32I R17, 0x3fe00000;                                     /* 0x18ff800000045de2 */
        /*0258*/                   DADD R24, -R24, R10;                                        /* 0x4800000029861e01 */
        /*0260*/                   DADD R26, -R26, R12;                                        /* 0x4800000031a69e01 */
        /*0268*/                   DMUL R12, R24, R24;                                         /* 0x5000000061831c01 */
        /*0270*/                   DFMA R10, R26, R26, R12;                                    /* 0x2018000069a29c01 */
        /*0278*/                   IADD32I R19, R11, -0x3500000;                               /* 0x0bf2c00000b4dc02 */
                                                                                               /* 0x2202923292928277 */
        /*0288*/                   MUFU.RSQ64H R13, R11;                                       /* 0xc80000001cb35c00 */
        /*0290*/                   MOV R12, R19;                                               /* 0x280000004c031de4 */
        /*0298*/                   DMUL R14, R12, R12;                                         /* 0x5000000030c39c01 */
        /*02a0*/                   DFMA R14, -R10, R14, c[0x2][0x0];                           /* 0x201c880000a39e01 */
        /*02a8*/                   DFMA R16, R14, 0.375, R16;                                  /* 0x2020cff600e41c01 */
        /*02b0*/                   DMUL R14, R12, R14;                                         /* 0x5000000038c39c01 */
        /*02b8*/                   DFMA R20, R16, R14, R12;                                    /* 0x2018000039051c01 */
                                                                                               /* 0x2202820280420287 */
        /*02c8*/                   ISETP.LT.U32.AND P1, PT, R19, c[0x2][0x8], PT;              /* 0x188e48002133dc03 */
        /*02d0*/                   DMUL R16, R10, R20;                                         /* 0x5000000050a41c01 */
        /*02d8*/                   IADD32I R23, R21, -0x100000;                                /* 0x0bffc0000155dc02 */
        /*02e0*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*02e8*/                   DFMA R12, -R16, R16, R10;                                   /* 0x2014000041031e01 */
        /*02f0*/                   MOV R15, R23;                                               /* 0x280000005c03dde4 */
        /*02f8*/                   DFMA R28, R12, R14, R16;                                    /* 0x2020000038c71c01 */
                                                                                               /* 0x22c2b2b2b20042b7 */
        /*0308*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0310*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*0318*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*0320*/                   MOV R19, R23;                                               /* 0x280000005c04dde4 */
        /*0328*/                   CAL 0x1a38;                                                 /* 0x5000005c20011c07 */
        /*0330*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0338*/                   IMAD R10.CC, R37, 0x8, R18;                                 /* 0x2025c00022529ca3 */
                                                                                               /* 0x2282804200428047 */
        /*0348*/                   IMAD.HI.X R11, R37, 0x8, R9;                                /* 0x2092c0002252dce3 */
        /*0350*/                   LD.E.64 R16, [R4+0x18];                                     /* 0x8400000060441ca5 */
        /*0358*/                   LD.E.64 R10, [R10+0x18];                                    /* 0x8400000060a29ca5 */
        /*0360*/                   DFMA R12, R28, R28, c[0x2][0x18];                           /* 0x2038880061c31c01 */
        /*0368*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
        /*0370*/                   SSY 0x418;                                                  /* 0x6000000280001c07 */
        /*0378*/                   MUFU.RCP64H R19, R13;                                       /* 0xc800000018d4dc00 */
                                                                                               /* 0x2292929293f29297 */
        /*0388*/                   DFMA R14, -R12, R18, c[0x2][0x0];                           /* 0x2024880000c39e01 */
        /*0390*/                   DFMA R20, R14, R14, R14;                                    /* 0x201c000038e51c01 */
        /*0398*/                   DFMA R18, R18, R20, R18;                                    /* 0x2024000051249c01 */
        /*03a0*/                   DMUL R14, R10, c[0x2][0x10];                                /* 0x5000480040a39c01 */
        /*03a8*/                   DMUL R10, R14, R16;                                         /* 0x5000000040e29c01 */
        /*03b0*/                   DMUL R30, R10, R18;                                         /* 0x5000000048a79c01 */
        /*03b8*/                   DFMA R16, -R12, R30, R10;                                   /* 0x2014000078c41e01 */
                                                                                               /* 0x2202b2b2c2828207 */
        /*03c8*/                   DFMA R30, R18, R16, R30;                                    /* 0x203c000041279c01 */
        /*03d0*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0xc], PT;                 /* 0x160f480030bfdc80 */
        /*03d8*/                   FFMA R9, RZ, R13, R31;                                      /* 0x303e000037f25c00 */
        /*03e0*/                   FSETP.GT.AND P1, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008093dc80 */
        /*03e8*/               @P1 BRA CC.NEU, 0x410;                                          /* 0x40000000800005a7 */
        /*03f0*/                   CAL 0x16e8;                                                 /* 0x5000004bc0011c07 */
        /*03f8*/                   MOV R30, R16;                                               /* 0x2800000040079de4 */
                                                                                               /* 0x225232928042b2b7 */
        /*0408*/                   MOV R31, R17;                                               /* 0x280000004407dde4 */
        /*0410*/                   MUFU.RCP64H.S R15, R29;                                     /* 0xc800000019d3dc10 */
        /*0418*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*0420*/                   SSY 0x4b0;                                                  /* 0x6000000220001c07 */
        /*0428*/                   DFMA R10, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c29e01 */
        /*0430*/                   DFMA R12, R10, R10, R10;                                    /* 0x2014000028a31c01 */
        /*0438*/                   DMUL R10, R26, R30;                                         /* 0x5000000079a29c01 */
                                                                                               /* 0x2282804202929297 */
        /*0448*/                   DFMA R12, R14, R12, R14;                                    /* 0x201c000030e31c01 */
        /*0450*/                   DMUL R16, R10, R12;                                         /* 0x5000000030a41c01 */
        /*0458*/                   DFMA R14, -R28, R16, R10;                                   /* 0x2014000041c39e01 */
        /*0460*/                   DFMA R16, R12, R14, R16;                                    /* 0x2020000038c41c01 */
        /*0468*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0xc], PT;                 /* 0x160f480030bfdc80 */
        /*0470*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*0478*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
                                                                                               /* 0x220042b2b2b042c7 */
        /*0488*/                   FSETP.GT.AND P1, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008093dc80 */
        /*0490*/                   MOV R13, R29;                                               /* 0x2800000074035de4 */
        /*0498*/               @P1 BRA CC.NEU, 0x4a8;                                          /* 0x40000000200005a7 */
        /*04a0*/                   CAL 0x16e8;                                                 /* 0x5000004900011c07 */
        /*04a8*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*04b0*/                   LD.E.64 R12, [R2+0x4e218];                                  /* 0x8400138860231ca5 */
        /*04b8*/                   MUFU.RCP64H R15, R29;                                       /* 0xc800000019d3dc00 */
                                                                                               /* 0x2292925232928047 */
        /*04c8*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*04d0*/                   SSY 0x570;                                                  /* 0x6000000260001c07 */
        /*04d8*/                   DFMA R10, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c29e01 */
        /*04e0*/                   DFMA R18, R10, R10, R10;                                    /* 0x2014000028a49c01 */
        /*04e8*/                   DMUL R10, R24, R30;                                         /* 0x5000000079829c01 */
        /*04f0*/                   DFMA R20, R14, R18, R14;                                    /* 0x201c000048e51c01 */
        /*04f8*/                   DMUL R14, R20, R10;                                         /* 0x5000000029439c01 */
                                                                                               /* 0x22c280429233f207 */
        /*0508*/                   DFMA R18, -R28, R14, R10;                                   /* 0x2014000039c49e01 */
        /*0510*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0xc], PT;                 /* 0x160f480030bfdc80 */
        /*0518*/                   DADD R12, R16, R12;                                         /* 0x4800000031031c01 */
        /*0520*/                   DFMA R16, R20, R18, R14;                                    /* 0x201c000049441c01 */
        /*0528*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*0530*/                   ST.E.64 [R2+0x4e218], R12;                                  /* 0x9400138860231ca5 */
        /*0538*/                   FSETP.GT.AND P1, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008093dc80 */
                                                                                               /* 0x2293f2b2b2b042b7 */
        /*0548*/               @P1 BRA CC.NEU, 0x568;                                          /* 0x40000000600005a7 */
        /*0550*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*0558*/                   MOV R13, R29;                                               /* 0x2800000074035de4 */
        /*0560*/                   CAL 0x16e8;                                                 /* 0x5000004600011c07 */
        /*0568*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0570*/                   LD.E.64 R10, [R2+0x5dc18];                                  /* 0x8400177060229ca5 */
        /*0578*/                   DADD R16, R16, R10;                                         /* 0x4800000029041c01 */
                                                                                               /* 0x228042b2c2c042b7 */
        /*0588*/                   ST.E.64.S [R2+0x5dc18], R16;                                /* 0x9400177060241cb5 */
        /*0590*/                   ISUB RZ.CC, R0, R33;                                        /* 0x48010000840fdd03 */
        /*0598*/                   SSY 0x9c0;                                                  /* 0x6000001080001c07 */
        /*05a0*/                   ISETP.EQ.X.AND P1, PT, R41, R36, PT;                        /* 0x190e00009293dc63 */
        /*05a8*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*05b0*/                   MOV32I R9, 0x3;                                             /* 0x180000000c025de2 */
        /*05b8*/                   SHL R10, R0, -0x1d;                                         /* 0x6000ffff8c029c03 */
                                                                                               /* 0x2202c232b0420047 */
        /*05c8*/                   IADD32I RZ.CC, R9, -0x20;                                   /* 0x0fffffff809fdc02 */
        /*05d0*/                   SHR.U32 R12, R0, 0x1d;                                      /* 0x5800c00074031c03 */
        /*05d8*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
        /*05e0*/                   SSY 0x760;                                                  /* 0x60000005e0001c07 */
        /*05e8*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
        /*05f0*/                   ISCADD R9.CC, R37, 0x8, 0x3;                                /* 0x4001c00022525c63 */
        /*05f8*/                   IMAD.HI.X R13, R37, 0x8, RZ;                                /* 0x20fec00022535ce3 */
                                                                                               /* 0x2280428042b04207 */
        /*0608*/                   IADD R20.CC, R9, R18;                                       /* 0x4801000048951c03 */
        /*0610*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*0618*/                   SHL R11, R41, 0x3;                                          /* 0x6000c0000e92dc03 */
        /*0620*/                   IADD.X R21, R13, R9;                                        /* 0x4800000024d55c43 */
        /*0628*/                   IMAD.U32.U32 R16.CC, R0, 0x8, R18;                          /* 0x2025c00020041c03 */
        /*0630*/               @P1 IADD R10, R11, R12;                                         /* 0x4800000030b28403 */
        /*0638*/                   LD.E.64 R24, [R20+0x1f410];                                 /* 0x840007d041461ca5 */
                                                                                               /* 0x2233f0420042e047 */
        /*0648*/                   IADD.X R17, R10, R9;                                        /* 0x4800000024a45c43 */
        /*0650*/                   LD.E.64 R26, [R20+0xfa10];                                  /* 0x840003e841469ca5 */
        /*0658*/                   LD.E.64 R10, [R16+0x1f410];                                 /* 0x840007d041029ca5 */
        /*0660*/                   MOV32I R20, 0x0;                                            /* 0x1800000000051de2 */
        /*0668*/                   LD.E.64 R12, [R16+0xfa10];                                  /* 0x840003e841031ca5 */
        /*0670*/                   MOV32I R21, 0x3fe00000;                                     /* 0x18ff800000055de2 */
        /*0678*/                   DADD R24, -R24, R10;                                        /* 0x4800000029861e01 */
                                                                                               /* 0x2292828042929257 */
        /*0688*/                   DADD R26, -R26, R12;                                        /* 0x4800000031a69e01 */
        /*0690*/                   DMUL R12, R24, R24;                                         /* 0x5000000061831c01 */
        /*0698*/                   DFMA R10, R26, R26, R12;                                    /* 0x2018000069a29c01 */
        /*06a0*/                   IADD32I R19, R11, -0x3500000;                               /* 0x0bf2c00000b4dc02 */
        /*06a8*/                   MUFU.RSQ64H R13, R11;                                       /* 0xc80000001cb35c00 */
        /*06b0*/                   MOV R12, R19;                                               /* 0x280000004c031de4 */
        /*06b8*/                   DMUL R14, R12, R12;                                         /* 0x5000000030c39c01 */
                                                                                               /* 0x2202028202923297 */
        /*06c8*/                   DFMA R14, -R10, R14, c[0x2][0x0];                           /* 0x201c880000a39e01 */
        /*06d0*/                   DFMA R16, R14, 0.375, R20;                                  /* 0x2028cff600e41c01 */
        /*06d8*/                   DMUL R14, R12, R14;                                         /* 0x5000000038c39c01 */
        /*06e0*/                   DFMA R20, R16, R14, R12;                                    /* 0x2018000039051c01 */
        /*06e8*/                   ISETP.LT.U32.AND P1, PT, R19, c[0x2][0x8], PT;              /* 0x188e48002133dc03 */
        /*06f0*/                   DMUL R16, R10, R20;                                         /* 0x5000000050a41c01 */
        /*06f8*/                   IADD32I R23, R21, -0x100000;                                /* 0x0bffc0000155dc02 */
                                                                                               /* 0x220042b202820277 */
        /*0708*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*0710*/                   DFMA R12, -R16, R16, R10;                                   /* 0x2014000041031e01 */
        /*0718*/                   MOV R15, R23;                                               /* 0x280000005c03dde4 */
        /*0720*/                   DFMA R28, R12, R14, R16;                                    /* 0x2020000038c71c01 */
        /*0728*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0730*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*0738*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
                                                                                               /* 0x228042c042b2b2b7 */
        /*0748*/                   MOV R19, R23;                                               /* 0x280000005c04dde4 */
        /*0750*/                   CAL 0x1a38;                                                 /* 0x5000004b80011c07 */
        /*0758*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0760*/                   IMAD R20.CC, R37, 0x8, R18;                                 /* 0x2025c00022551ca3 */
        /*0768*/                   LD.E.64 R14, [R4+0x20];                                     /* 0x8400000080439ca5 */
        /*0770*/                   IMAD.HI.X R21, R37, 0x8, R9;                                /* 0x2092c00022555ce3 */
        /*0778*/                   MOV32I R16, 0x1;                                            /* 0x1800000004041de2 */
                                                                                               /* 0x23f2929280429047 */
        /*0788*/                   LD.E.64 R10, [R20+0x18];                                    /* 0x8400000061429ca5 */
        /*0790*/                   DFMA R12, R28, R28, c[0x2][0x18];                           /* 0x2038880061c31c01 */
        /*0798*/                   MUFU.RCP64H R17, R13;                                       /* 0xc800000018d45c00 */
        /*07a0*/                   SSY 0x850;                                                  /* 0x60000002a0001c07 */
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
                                                                                               /* 0x22b042b2b042b2c7 */
        /*0808*/                   FSETP.GT.AND P1, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008093dc80 */
        /*0810*/               @P1 BRA CC.NEU, 0x840;                                          /* 0x40000000a00005a7 */
        /*0818*/                   MOV R10, R18;                                               /* 0x2800000048029de4 */
        /*0820*/                   MOV R11, R19;                                               /* 0x280000004c02dde4 */
        /*0828*/                   CAL 0x16e8;                                                 /* 0x5000003ae0011c07 */
        /*0830*/                   MOV R30, R16;                                               /* 0x2800000040079de4 */
        /*0838*/                   MOV R31, R17;                                               /* 0x280000004407dde4 */
                                                                                               /* 0x22925232928042b7 */
        /*0848*/                   MUFU.RCP64H.S R15, R29;                                     /* 0xc800000019d3dc10 */
        /*0850*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*0858*/                   SSY 0x8e8;                                                  /* 0x6000000220001c07 */
        /*0860*/                   DFMA R10, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c29e01 */
        /*0868*/                   DFMA R12, R10, R10, R10;                                    /* 0x2014000028a31c01 */
        /*0870*/                   DMUL R10, R26, R30;                                         /* 0x5000000079a29c01 */
        /*0878*/                   DFMA R12, R14, R12, R14;                                    /* 0x201c000030e31c01 */
                                                                                               /* 0x22c2828042029297 */
        /*0888*/                   DMUL R16, R10, R12;                                         /* 0x5000000030a41c01 */
        /*0890*/                   DFMA R14, -R28, R16, R10;                                   /* 0x2014000041c39e01 */
        /*0898*/                   DFMA R16, R12, R14, R16;                                    /* 0x2020000038c41c01 */
        /*08a0*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0xc], PT;                 /* 0x160f480030bfdc80 */
        /*08a8*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*08b0*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*08b8*/                   FSETP.GT.AND P1, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008093dc80 */
                                                                                               /* 0x22820042b2b2b047 */
        /*08c8*/                   MOV R13, R29;                                               /* 0x2800000074035de4 */
        /*08d0*/               @P1 BRA CC.NEU, 0x8e0;                                          /* 0x40000000200005a7 */
        /*08d8*/                   CAL 0x16e8;                                                 /* 0x5000003820011c07 */
        /*08e0*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*08e8*/                   LD.E.64 R12, [R2+0x4e218];                                  /* 0x8400138860231ca5 */
        /*08f0*/                   MUFU.RCP64H R15, R29;                                       /* 0xc800000019d3dc00 */
        /*08f8*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
                                                                                               /* 0x2202029292523297 */
        /*0908*/                   DFMA R10, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c29e01 */
        /*0910*/                   DFMA R18, R10, R10, R10;                                    /* 0x2014000028a49c01 */
        /*0918*/                   DMUL R10, R24, R30;                                         /* 0x5000000079829c01 */
        /*0920*/                   DFMA R20, R14, R18, R14;                                    /* 0x201c000048e51c01 */
        /*0928*/                   DMUL R14, R20, R10;                                         /* 0x5000000029439c01 */
        /*0930*/                   DFMA R18, -R28, R14, R10;                                   /* 0x2014000039c49e01 */
        /*0938*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0xc], PT;                 /* 0x160f480030bfdc80 */
                                                                                               /* 0x22b2c280429233f7 */
        /*0948*/                   SSY 0x9a8;                                                  /* 0x6000000160001c07 */
        /*0950*/                   DADD R12, R16, R12;                                         /* 0x4800000031031c01 */
        /*0958*/                   DFMA R16, R20, R18, R14;                                    /* 0x201c000049441c01 */
        /*0960*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*0968*/                   ST.E.64 [R2+0x4e218], R12;                                  /* 0x9400138860231ca5 */
        /*0970*/                   FSETP.GT.AND P1, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008093dc80 */
        /*0978*/               @P1 BRA CC.NEU, 0x9a0;                                          /* 0x40000000800005a7 */
                                                                                               /* 0x22b293f2b2b2b047 */
        /*0988*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*0990*/                   MOV R13, R29;                                               /* 0x2800000074035de4 */
        /*0998*/                   CAL 0x16e8;                                                 /* 0x5000003520011c07 */
        /*09a0*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*09a8*/                   LD.E.64 R10, [R2+0x5dc18];                                  /* 0x8400177060229ca5 */
        /*09b0*/                   DADD R16, R16, R10;                                         /* 0x4800000029041c01 */
        /*09b8*/                   ST.E.64.S [R2+0x5dc18], R16;                                /* 0x9400177060241cb5 */
                                                                                               /* 0x22028042b2c2c047 */
        /*09c8*/                   ISUB RZ.CC, R6, R33;                                        /* 0x48010000846fdd03 */
        /*09d0*/                   SSY 0xdf8;                                                  /* 0x6000001080001c07 */
        /*09d8*/                   ISETP.EQ.X.AND P1, PT, R40, R36, PT;                        /* 0x190e00009283dc63 */
        /*09e0*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*09e8*/                   MOV32I R9, 0x3;                                             /* 0x180000000c025de2 */
        /*09f0*/                   SHL R10, R6, -0x1d;                                         /* 0x6000ffff8c629c03 */
        /*09f8*/                   IADD32I RZ.CC, R9, -0x20;                                   /* 0x0fffffff809fdc02 */
                                                                                               /* 0x220042c23042b047 */
        /*0a08*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
        /*0a10*/                   SHL R11, R40, 0x3;                                          /* 0x6000c0000e82dc03 */
        /*0a18*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
        /*0a20*/                   SSY 0xb98;                                                  /* 0x60000005c0001c07 */
        /*0a28*/                   ISCADD R9.CC, R37, 0x8, 0x3;                                /* 0x4001c00022525c63 */
        /*0a30*/                   IMAD.HI.X R13, R37, 0x8, RZ;                                /* 0x20fec00022535ce3 */
        /*0a38*/                   IADD R20.CC, R9, R18;                                       /* 0x4801000048951c03 */
                                                                                               /* 0x228280428042b047 */
        /*0a48*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*0a50*/                   SHR.U32 R12, R6, 0x1d;                                      /* 0x5800c00074631c03 */
        /*0a58*/                   IADD.X R21, R13, R9;                                        /* 0x4800000024d55c43 */
        /*0a60*/                   IMAD.U32.U32 R16.CC, R6, 0x8, R18;                          /* 0x2025c00020641c03 */
        /*0a68*/               @P1 IADD R10, R11, R12;                                         /* 0x4800000030b28403 */
        /*0a70*/                   LD.E.64 R24, [R20+0x1f410];                                 /* 0x840007d041461ca5 */
        /*0a78*/                   IADD.X R17, R10, R9;                                        /* 0x4800000024a45c43 */
                                                                                               /* 0x223273f042a23237 */
        /*0a88*/                   LD.E.64 R10, [R16+0x1f410];                                 /* 0x840007d041029ca5 */
        /*0a90*/                   LD.E.64 R26, [R20+0xfa10];                                  /* 0x840003e841469ca5 */
        /*0a98*/                   LD.E.64 R12, [R16+0xfa10];                                  /* 0x840003e841031ca5 */
        /*0aa0*/                   MOV32I R20, 0x0;                                            /* 0x1800000000051de2 */
        /*0aa8*/                   MOV32I R21, 0x3fe00000;                                     /* 0x18ff800000055de2 */
        /*0ab0*/                   DADD R24, -R24, R10;                                        /* 0x4800000029861e01 */
        /*0ab8*/                   DADD R26, -R26, R12;                                        /* 0x4800000031a69e01 */
                                                                                               /* 0x2292928280429297 */
        /*0ac8*/                   DMUL R12, R24, R24;                                         /* 0x5000000061831c01 */
        /*0ad0*/                   DFMA R10, R26, R26, R12;                                    /* 0x2018000069a29c01 */
        /*0ad8*/                   IADD32I R19, R11, -0x3500000;                               /* 0x0bf2c00000b4dc02 */
        /*0ae0*/                   MUFU.RSQ64H R13, R11;                                       /* 0xc80000001cb35c00 */
        /*0ae8*/                   MOV R12, R19;                                               /* 0x280000004c031de4 */
        /*0af0*/                   DMUL R14, R12, R12;                                         /* 0x5000000030c39c01 */
        /*0af8*/                   DFMA R14, -R10, R14, c[0x2][0x0];                           /* 0x201c880000a39e01 */
                                                                                               /* 0x2280420282029237 */
        /*0b08*/                   DFMA R16, R14, 0.375, R20;                                  /* 0x2028cff600e41c01 */
        /*0b10*/                   DMUL R14, R12, R14;                                         /* 0x5000000038c39c01 */
        /*0b18*/                   DFMA R20, R16, R14, R12;                                    /* 0x2018000039051c01 */
        /*0b20*/                   ISETP.LT.U32.AND P1, PT, R19, c[0x2][0x8], PT;              /* 0x188e48002133dc03 */
        /*0b28*/                   DMUL R16, R10, R20;                                         /* 0x5000000050a41c01 */
        /*0b30*/                   IADD32I R23, R21, -0x100000;                                /* 0x0bffc0000155dc02 */
        /*0b38*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
                                                                                               /* 0x22b20042b2028207 */
        /*0b48*/                   DFMA R12, -R16, R16, R10;                                   /* 0x2014000041031e01 */
        /*0b50*/                   MOV R15, R23;                                               /* 0x280000005c03dde4 */
        /*0b58*/                   DFMA R28, R12, R14, R16;                                    /* 0x2020000038c71c01 */
        /*0b60*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0b68*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*0b70*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*0b78*/                   MOV R19, R23;                                               /* 0x280000005c04dde4 */
                                                                                               /* 0x22028042c042b2b7 */
        /*0b88*/                   CAL 0x1a38;                                                 /* 0x5000003aa0011c07 */
        /*0b90*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0b98*/                   IMAD R20.CC, R37, 0x8, R18;                                 /* 0x2025c00022551ca3 */
        /*0ba0*/                   LD.E.64 R14, [R4+0x28];                                     /* 0x84000000a0439ca5 */
        /*0ba8*/                   IMAD.HI.X R21, R37, 0x8, R9;                                /* 0x2092c00022555ce3 */
        /*0bb0*/                   MOV32I R16, 0x1;                                            /* 0x1800000004041de2 */
        /*0bb8*/                   LD.E.64 R10, [R20+0x18];                                    /* 0x8400000061429ca5 */
                                                                                               /* 0x2293f29292804297 */
        /*0bc8*/                   DFMA R12, R28, R28, c[0x2][0x18];                           /* 0x2038880061c31c01 */
        /*0bd0*/                   MUFU.RCP64H R17, R13;                                       /* 0xc800000018d45c00 */
        /*0bd8*/                   SSY 0xc80;                                                  /* 0x6000000280001c07 */
        /*0be0*/                   DFMA R18, -R12, R16, c[0x2][0x0];                           /* 0x2020880000c49e01 */
        /*0be8*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*0bf0*/                   DFMA R16, R16, R18, R16;                                    /* 0x2020000049041c01 */
        /*0bf8*/                   DMUL R10, R10, c[0x2][0x10];                                /* 0x5000480040a29c01 */
                                                                                               /* 0x22c2828202929297 */
        /*0c08*/                   DMUL R18, R10, R14;                                         /* 0x5000000038a49c01 */
        /*0c10*/                   DMUL R30, R18, R16;                                         /* 0x5000000041279c01 */
        /*0c18*/                   DFMA R10, -R12, R30, R18;                                   /* 0x2024000078c29e01 */
        /*0c20*/                   DFMA R30, R16, R10, R30;                                    /* 0x203c000029079c01 */
        /*0c28*/                   FSET.GTU.AND RZ.CC, |R19|, c[0x2][0xc], PT;                 /* 0x160f4800313fdc80 */
        /*0c30*/                   FFMA R9, RZ, R13, R31;                                      /* 0x303e000037f25c00 */
        /*0c38*/                   FSETP.GT.AND P1, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008093dc80 */
                                                                                               /* 0x22b2b042b2b042b7 */
        /*0c48*/               @P1 BRA CC.NEU, 0xc78;                                          /* 0x40000000a00005a7 */
        /*0c50*/                   MOV R10, R18;                                               /* 0x2800000048029de4 */
        /*0c58*/                   MOV R11, R19;                                               /* 0x280000004c02dde4 */
        /*0c60*/                   CAL 0x16e8;                                                 /* 0x5000002a00011c07 */
        /*0c68*/                   MOV R30, R16;                                               /* 0x2800000040079de4 */
        /*0c70*/                   MOV R31, R17;                                               /* 0x280000004407dde4 */
        /*0c78*/                   MUFU.RCP64H.S R15, R29;                                     /* 0xc800000019d3dc10 */
                                                                                               /* 0x2292925232928047 */
        /*0c88*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*0c90*/                   SSY 0xd20;                                                  /* 0x6000000220001c07 */
        /*0c98*/                   DFMA R10, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c29e01 */
        /*0ca0*/                   DFMA R12, R10, R10, R10;                                    /* 0x2014000028a31c01 */
        /*0ca8*/                   DMUL R10, R26, R30;                                         /* 0x5000000079a29c01 */
        /*0cb0*/                   DFMA R12, R14, R12, R14;                                    /* 0x201c000030e31c01 */
        /*0cb8*/                   DMUL R16, R10, R12;                                         /* 0x5000000030a41c01 */
                                                                                               /* 0x22b2028280420297 */
        /*0cc8*/                   DFMA R14, -R28, R16, R10;                                   /* 0x2014000041c39e01 */
        /*0cd0*/                   DFMA R16, R12, R14, R16;                                    /* 0x2020000038c41c01 */
        /*0cd8*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0xc], PT;                 /* 0x160f480030bfdc80 */
        /*0ce0*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*0ce8*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*0cf0*/                   FSETP.GT.AND P1, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008093dc80 */
        /*0cf8*/                   MOV R13, R29;                                               /* 0x2800000074035de4 */
                                                                                               /* 0x2280420042b2b2b7 */
        /*0d08*/               @P1 BRA CC.NEU, 0xd18;                                          /* 0x40000000200005a7 */
        /*0d10*/                   CAL 0x16e8;                                                 /* 0x5000002740011c07 */
        /*0d18*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0d20*/                   LD.E.64 R12, [R2+0x4e218];                                  /* 0x8400138860231ca5 */
        /*0d28*/                   MUFU.RCP64H R15, R29;                                       /* 0xc800000019d3dc00 */
        /*0d30*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*0d38*/                   SSY 0xde0;                                                  /* 0x6000000280001c07 */
                                                                                               /* 0x23f2029292523297 */
        /*0d48*/                   DFMA R10, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c29e01 */
        /*0d50*/                   DFMA R18, R10, R10, R10;                                    /* 0x2014000028a49c01 */
        /*0d58*/                   DMUL R10, R24, R30;                                         /* 0x5000000079829c01 */
        /*0d60*/                   DFMA R20, R14, R18, R14;                                    /* 0x201c000048e51c01 */
        /*0d68*/                   DMUL R14, R20, R10;                                         /* 0x5000000029439c01 */
        /*0d70*/                   DFMA R18, -R28, R14, R10;                                   /* 0x2014000039c49e01 */
        /*0d78*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0xc], PT;                 /* 0x160f480030bfdc80 */
                                                                                               /* 0x2202b2c280429237 */
        /*0d88*/                   DADD R12, R16, R12;                                         /* 0x4800000031031c01 */
        /*0d90*/                   DFMA R16, R20, R18, R14;                                    /* 0x201c000049441c01 */
        /*0d98*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*0da0*/                   ST.E.64 [R2+0x4e218], R12;                                  /* 0x9400138860231ca5 */
        /*0da8*/                   FSETP.GT.AND P1, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008093dc80 */
        /*0db0*/               @P1 BRA CC.NEU, 0xdd8;                                          /* 0x40000000800005a7 */
        /*0db8*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
                                                                                               /* 0x22c2b293f2b2b2b7 */
        /*0dc8*/                   MOV R13, R29;                                               /* 0x2800000074035de4 */
        /*0dd0*/                   CAL 0x16e8;                                                 /* 0x5000002440011c07 */
        /*0dd8*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0de0*/                   LD.E.64 R10, [R2+0x5dc18];                                  /* 0x8400177060229ca5 */
        /*0de8*/                   DADD R16, R16, R10;                                         /* 0x4800000029041c01 */
        /*0df0*/                   ST.E.64.S [R2+0x5dc18], R16;                                /* 0x9400177060241cb5 */
        /*0df8*/                   ISUB RZ.CC, R7, R33;                                        /* 0x48010000847fdd03 */
                                                                                               /* 0x2200428042b2c047 */
        /*0e08*/                   ISETP.EQ.X.AND P1, PT, R39, R36, PT;                        /* 0x190e00009273dc63 */
        /*0e10*/                   SSY 0x1230;                                                 /* 0x6000001060001c07 */
        /*0e18*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0e20*/                   MOV32I R9, 0x3;                                             /* 0x180000000c025de2 */
        /*0e28*/                   SHL R10, R7, -0x1d;                                         /* 0x6000ffff8c729c03 */
        /*0e30*/                   IADD32I RZ.CC, R9, -0x20;                                   /* 0x0fffffff809fdc02 */
        /*0e38*/                   SHL R11, R39, 0x3;                                          /* 0x6000c0000e72dc03 */
                                                                                               /* 0x22020042c232b047 */
        /*0e48*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
        /*0e50*/                   SSY 0xfd0;                                                  /* 0x60000005e0001c07 */
        /*0e58*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
        /*0e60*/                   ISCADD R9.CC, R37, 0x8, 0x3;                                /* 0x4001c00022525c63 */
        /*0e68*/                   IMAD.HI.X R13, R37, 0x8, RZ;                                /* 0x20fec00022535ce3 */
        /*0e70*/                   IADD R20.CC, R9, R18;                                       /* 0x4801000048951c03 */
        /*0e78*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
                                                                                               /* 0x22328042c04202a7 */
        /*0e88*/                   SHR.U32 R12, R7, 0x1d;                                      /* 0x5800c00074731c03 */
        /*0e90*/                   IADD.X R21, R13, R9;                                        /* 0x4800000024d55c43 */
        /*0e98*/                   IMAD.U32.U32 R16.CC, R7, 0x8, R18;                          /* 0x2025c00020741c03 */
        /*0ea0*/               @P1 IADD R10, R11, R12;                                         /* 0x4800000030b28403 */
        /*0ea8*/                   IADD.X R17, R10, R9;                                        /* 0x4800000024a45c43 */
        /*0eb0*/                   LD.E.64 R24, [R20+0x1f410];                                 /* 0x840007d041461ca5 */
        /*0eb8*/                   LD.E.64 R10, [R16+0x1f410];                                 /* 0x840007d041029ca5 */
                                                                                               /* 0x22923273f042a237 */
        /*0ec8*/                   LD.E.64 R26, [R20+0xfa10];                                  /* 0x840003e841469ca5 */
        /*0ed0*/                   LD.E.64 R12, [R16+0xfa10];                                  /* 0x840003e841031ca5 */
        /*0ed8*/                   MOV32I R20, 0x0;                                            /* 0x1800000000051de2 */
        /*0ee0*/                   MOV32I R21, 0x3fe00000;                                     /* 0x18ff800000055de2 */
        /*0ee8*/                   DADD R24, -R24, R10;                                        /* 0x4800000029861e01 */
        /*0ef0*/                   DADD R26, -R26, R12;                                        /* 0x4800000031a69e01 */
        /*0ef8*/                   DMUL R12, R24, R24;                                         /* 0x5000000061831c01 */
                                                                                               /* 0x2232929282804297 */
        /*0f08*/                   DFMA R10, R26, R26, R12;                                    /* 0x2018000069a29c01 */
        /*0f10*/                   IADD32I R19, R11, -0x3500000;                               /* 0x0bf2c00000b4dc02 */
        /*0f18*/                   MUFU.RSQ64H R13, R11;                                       /* 0xc80000001cb35c00 */
        /*0f20*/                   MOV R12, R19;                                               /* 0x280000004c031de4 */
        /*0f28*/                   DMUL R14, R12, R12;                                         /* 0x5000000030c39c01 */
        /*0f30*/                   DFMA R14, -R10, R14, c[0x2][0x0];                           /* 0x201c880000a39e01 */
        /*0f38*/                   DFMA R16, R14, 0.375, R20;                                  /* 0x2028cff600e41c01 */
                                                                                               /* 0x2202804202820297 */
        /*0f48*/                   DMUL R14, R12, R14;                                         /* 0x5000000038c39c01 */
        /*0f50*/                   DFMA R20, R16, R14, R12;                                    /* 0x2018000039051c01 */
        /*0f58*/                   ISETP.LT.U32.AND P1, PT, R19, c[0x2][0x8], PT;              /* 0x188e48002133dc03 */
        /*0f60*/                   DMUL R16, R10, R20;                                         /* 0x5000000050a41c01 */
        /*0f68*/                   IADD32I R23, R21, -0x100000;                                /* 0x0bffc0000155dc02 */
        /*0f70*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*0f78*/                   DFMA R12, -R16, R16, R10;                                   /* 0x2014000041031e01 */
                                                                                               /* 0x22b2b20042b20287 */
        /*0f88*/                   MOV R15, R23;                                               /* 0x280000005c03dde4 */
        /*0f90*/                   DFMA R28, R12, R14, R16;                                    /* 0x2020000038c71c01 */
        /*0f98*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0fa0*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*0fa8*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*0fb0*/                   MOV R19, R23;                                               /* 0x280000005c04dde4 */
        /*0fb8*/                   CAL 0x1a38;                                                 /* 0x50000029e0011c07 */
                                                                                               /* 0x2290428042c042b7 */
        /*0fc8*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0fd0*/                   IMAD R20.CC, R37, 0x8, R18;                                 /* 0x2025c00022551ca3 */
        /*0fd8*/                   LD.E.64 R14, [R4+0x30];                                     /* 0x84000000c0439ca5 */
        /*0fe0*/                   IMAD.HI.X R21, R37, 0x8, R9;                                /* 0x2092c00022555ce3 */
        /*0fe8*/                   MOV32I R16, 0x1;                                            /* 0x1800000004041de2 */
        /*0ff0*/                   LD.E.64 R10, [R20+0x18];                                    /* 0x8400000061429ca5 */
        /*0ff8*/                   DFMA R12, R28, R28, c[0x2][0x18];                           /* 0x2038880061c31c01 */
                                                                                               /* 0x229293f292928047 */
        /*1008*/                   MUFU.RCP64H R17, R13;                                       /* 0xc800000018d45c00 */
        /*1010*/                   SSY 0x10b8;                                                 /* 0x6000000280001c07 */
        /*1018*/                   DFMA R18, -R12, R16, c[0x2][0x0];                           /* 0x2020880000c49e01 */
        /*1020*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*1028*/                   DFMA R16, R16, R18, R16;                                    /* 0x2020000049041c01 */
        /*1030*/                   DMUL R10, R10, c[0x2][0x10];                                /* 0x5000480040a29c01 */
        /*1038*/                   DMUL R18, R10, R14;                                         /* 0x5000000038a49c01 */
                                                                                               /* 0x22b2c28282029297 */
        /*1048*/                   DMUL R30, R18, R16;                                         /* 0x5000000041279c01 */
        /*1050*/                   DFMA R10, -R12, R30, R18;                                   /* 0x2024000078c29e01 */
        /*1058*/                   DFMA R30, R16, R10, R30;                                    /* 0x203c000029079c01 */
        /*1060*/                   FSET.GTU.AND RZ.CC, |R19|, c[0x2][0xc], PT;                 /* 0x160f4800313fdc80 */
        /*1068*/                   FFMA R9, RZ, R13, R31;                                      /* 0x303e000037f25c00 */
        /*1070*/                   FSETP.GT.AND P1, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008093dc80 */
        /*1078*/               @P1 BRA CC.NEU, 0x10b0;                                         /* 0x40000000c00005a7 */
                                                                                               /* 0x2282b2b042b2b047 */
        /*1088*/                   MOV R10, R18;                                               /* 0x2800000048029de4 */
        /*1090*/                   MOV R11, R19;                                               /* 0x280000004c02dde4 */
        /*1098*/                   CAL 0x16e8;                                                 /* 0x5000001920011c07 */
        /*10a0*/                   MOV R30, R16;                                               /* 0x2800000040079de4 */
        /*10a8*/                   MOV R31, R17;                                               /* 0x280000004407dde4 */
        /*10b0*/                   MUFU.RCP64H.S R15, R29;                                     /* 0xc800000019d3dc10 */
        /*10b8*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
                                                                                               /* 0x2202929292523297 */
        /*10c8*/                   DFMA R10, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c29e01 */
        /*10d0*/                   DFMA R12, R10, R10, R10;                                    /* 0x2014000028a31c01 */
        /*10d8*/                   DMUL R10, R26, R30;                                         /* 0x5000000079a29c01 */
        /*10e0*/                   DFMA R12, R14, R12, R14;                                    /* 0x201c000030e31c01 */
        /*10e8*/                   DMUL R16, R10, R12;                                         /* 0x5000000030a41c01 */
        /*10f0*/                   DFMA R14, -R28, R16, R10;                                   /* 0x2014000041c39e01 */
        /*10f8*/                   DFMA R16, R12, R14, R16;                                    /* 0x2020000038c41c01 */
                                                                                               /* 0x22b042c280428047 */
        /*1108*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0xc], PT;                 /* 0x160f480030bfdc80 */
        /*1110*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*1118*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*1120*/                   SSY 0x1158;                                                 /* 0x60000000c0001c07 */
        /*1128*/                   FSETP.GT.AND P1, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008093dc80 */
        /*1130*/                   MOV R13, R29;                                               /* 0x2800000074035de4 */
        /*1138*/               @P1 BRA CC.NEU, 0x1150;                                         /* 0x40000000400005a7 */
                                                                                               /* 0x229280420042b2b7 */
        /*1148*/                   CAL 0x16e8;                                                 /* 0x5000001660011c07 */
        /*1150*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*1158*/                   LD.E.64 R12, [R2+0x4e218];                                  /* 0x8400138860231ca5 */
        /*1160*/                   MUFU.RCP64H R15, R29;                                       /* 0xc800000019d3dc00 */
        /*1168*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*1170*/                   SSY 0x1218;                                                 /* 0x6000000280001c07 */
        /*1178*/                   DFMA R10, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c29e01 */
                                                                                               /* 0x2233f20292925237 */
        /*1188*/                   DFMA R18, R10, R10, R10;                                    /* 0x2014000028a49c01 */
        /*1190*/                   DMUL R10, R24, R30;                                         /* 0x5000000079829c01 */
        /*1198*/                   DFMA R20, R14, R18, R14;                                    /* 0x201c000048e51c01 */
        /*11a0*/                   DMUL R14, R20, R10;                                         /* 0x5000000029439c01 */
        /*11a8*/                   DFMA R18, -R28, R14, R10;                                   /* 0x2014000039c49e01 */
        /*11b0*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0xc], PT;                 /* 0x160f480030bfdc80 */
        /*11b8*/                   DADD R12, R16, R12;                                         /* 0x4800000031031c01 */
                                                                                               /* 0x22b042b2c2804297 */
        /*11c8*/                   DFMA R16, R20, R18, R14;                                    /* 0x201c000049441c01 */
        /*11d0*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*11d8*/                   ST.E.64 [R2+0x4e218], R12;                                  /* 0x9400138860231ca5 */
        /*11e0*/                   FSETP.GT.AND P1, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008093dc80 */
        /*11e8*/               @P1 BRA CC.NEU, 0x1210;                                         /* 0x40000000800005a7 */
        /*11f0*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*11f8*/                   MOV R13, R29;                                               /* 0x2800000074035de4 */
                                                                                               /* 0x22c042b293f2b2b7 */
        /*1208*/                   CAL 0x16e8;                                                 /* 0x5000001360011c07 */
        /*1210*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*1218*/                   LD.E.64 R10, [R2+0x5dc18];                                  /* 0x8400177060229ca5 */
        /*1220*/                   DADD R16, R16, R10;                                         /* 0x4800000029041c01 */
        /*1228*/                   ST.E.64.S [R2+0x5dc18], R16;                                /* 0x9400177060241cb5 */
        /*1230*/                   ISUB RZ.CC, R8, R33;                                        /* 0x48010000848fdd03 */
        /*1238*/                   SSY 0x1668;                                                 /* 0x60000010a0001c07 */
                                                                                               /* 0x22b200428042b2c7 */
        /*1248*/                   ISETP.EQ.X.AND P1, PT, R38, R36, PT;                        /* 0x190e00009263dc63 */
        /*1250*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*1258*/                   MOV32I R9, 0x3;                                             /* 0x180000000c025de2 */
        /*1260*/                   SHL R10, R8, -0x1d;                                         /* 0x6000ffff8c829c03 */
        /*1268*/                   IADD32I RZ.CC, R9, -0x20;                                   /* 0x0fffffff809fdc02 */
        /*1270*/                   SHL R11, R38, 0x3;                                          /* 0x6000c0000e62dc03 */
        /*1278*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
                                                                                               /* 0x22b0420042c23047 */
        /*1288*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
        /*1290*/                   SSY 0x1400;                                                 /* 0x60000005a0001c07 */
        /*1298*/                   ISCADD R9.CC, R37, 0x8, 0x3;                                /* 0x4001c00022525c63 */
        /*12a0*/                   IMAD.HI.X R13, R37, 0x8, RZ;                                /* 0x20fec00022535ce3 */
        /*12a8*/                   IADD R20.CC, R9, R18;                                       /* 0x4801000048951c03 */
        /*12b0*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*12b8*/                   SHR.U32 R12, R8, 0x1d;                                      /* 0x5800c00074831c03 */
                                                                                               /* 0x2252804280428047 */
        /*12c8*/                   IADD.X R21, R13, R9;                                        /* 0x4800000024d55c43 */
        /*12d0*/                   IMAD.U32.U32 R16.CC, R8, 0x8, R18;                          /* 0x2025c00020841c03 */
        /*12d8*/               @P1 IADD R10, R11, R12;                                         /* 0x4800000030b28403 */
        /*12e0*/                   LD.E.64 R24, [R20+0x1f410];                                 /* 0x840007d041461ca5 */
        /*12e8*/                   IADD.X R17, R10, R9;                                        /* 0x4800000024a45c43 */
        /*12f0*/                   LD.E.64 R26, [R20+0xfa10];                                  /* 0x840003e841469ca5 */
        /*12f8*/                   LD.E.64 R10, [R16+0x1f410];                                 /* 0x840007d041029ca5 */
                                                                                               /* 0x2292923253f20047 */
        /*1308*/                   LD.E.64 R12, [R16+0xfa10];                                  /* 0x840003e841031ca5 */
        /*1310*/                   MOV32I R20, 0x0;                                            /* 0x1800000000051de2 */
        /*1318*/                   MOV32I R21, 0x3fe00000;                                     /* 0x18ff800000055de2 */
        /*1320*/                   DADD R24, -R24, R10;                                        /* 0x4800000029861e01 */
        /*1328*/                   DADD R26, -R26, R12;                                        /* 0x4800000031a69e01 */
        /*1330*/                   DMUL R12, R24, R24;                                         /* 0x5000000061831c01 */
        /*1338*/                   DFMA R10, R26, R26, R12;                                    /* 0x2018000069a29c01 */
                                                                                               /* 0x2292329292828047 */
        /*1348*/                   IADD32I R19, R11, -0x3500000;                               /* 0x0bf2c00000b4dc02 */
        /*1350*/                   MUFU.RSQ64H R13, R11;                                       /* 0xc80000001cb35c00 */
        /*1358*/                   MOV R12, R19;                                               /* 0x280000004c031de4 */
        /*1360*/                   DMUL R14, R12, R12;                                         /* 0x5000000030c39c01 */
        /*1368*/                   DFMA R14, -R10, R14, c[0x2][0x0];                           /* 0x201c880000a39e01 */
        /*1370*/                   DFMA R16, R14, 0.375, R20;                                  /* 0x2028cff600e41c01 */
        /*1378*/                   DMUL R14, R12, R14;                                         /* 0x5000000038c39c01 */
                                                                                               /* 0x2282028042028207 */
        /*1388*/                   DFMA R20, R16, R14, R12;                                    /* 0x2018000039051c01 */
        /*1390*/                   ISETP.LT.U32.AND P1, PT, R19, c[0x2][0x8], PT;              /* 0x188e48002133dc03 */
        /*1398*/                   DMUL R16, R10, R20;                                         /* 0x5000000050a41c01 */
        /*13a0*/                   IADD32I R23, R21, -0x100000;                                /* 0x0bffc0000155dc02 */
        /*13a8*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*13b0*/                   DFMA R12, -R16, R16, R10;                                   /* 0x2014000041031e01 */
        /*13b8*/                   MOV R15, R23;                                               /* 0x280000005c03dde4 */
                                                                                               /* 0x22b2b2b20042b207 */
        /*13c8*/                   DFMA R28, R12, R14, R16;                                    /* 0x2020000038c71c01 */
        /*13d0*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*13d8*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*13e0*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*13e8*/                   MOV R19, R23;                                               /* 0x280000005c04dde4 */
        /*13f0*/                   CAL 0x1a38;                                                 /* 0x5000001900011c07 */
        /*13f8*/                   NOP.S;                                                      /* 0x4000000000001df4 */
                                                                                               /* 0x228290428042c047 */
        /*1408*/                   IMAD R20.CC, R37, 0x8, R18;                                 /* 0x2025c00022551ca3 */
        /*1410*/                   LD.E.64 R14, [R4+0x38];                                     /* 0x84000000e0439ca5 */
        /*1418*/                   IMAD.HI.X R21, R37, 0x8, R9;                                /* 0x2092c00022555ce3 */
        /*1420*/                   MOV32I R16, 0x1;                                            /* 0x1800000004041de2 */
        /*1428*/                   LD.E.64 R10, [R20+0x18];                                    /* 0x8400000061429ca5 */
        /*1430*/                   DFMA R12, R28, R28, c[0x2][0x18];                           /* 0x2038880061c31c01 */
        /*1438*/                   MUFU.RCP64H R17, R13;                                       /* 0xc800000018d45c00 */
                                                                                               /* 0x22929293f2029297 */
        /*1448*/                   DFMA R18, -R12, R16, c[0x2][0x0];                           /* 0x2020880000c49e01 */
        /*1450*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*1458*/                   DFMA R16, R16, R18, R16;                                    /* 0x2020000049041c01 */
        /*1460*/                   SSY 0x14f0;                                                 /* 0x6000000220001c07 */
        /*1468*/                   DMUL R10, R10, c[0x2][0x10];                                /* 0x5000480040a29c01 */
        /*1470*/                   DMUL R18, R10, R14;                                         /* 0x5000000038a49c01 */
        /*1478*/                   DMUL R30, R18, R16;                                         /* 0x5000000041279c01 */
                                                                                               /* 0x2202b2c282820297 */
        /*1488*/                   DFMA R10, -R12, R30, R18;                                   /* 0x2024000078c29e01 */
        /*1490*/                   DFMA R30, R16, R10, R30;                                    /* 0x203c000029079c01 */
        /*1498*/                   FSET.GTU.AND RZ.CC, |R19|, c[0x2][0xc], PT;                 /* 0x160f4800313fdc80 */
        /*14a0*/                   FFMA R9, RZ, R13, R31;                                      /* 0x303e000037f25c00 */
        /*14a8*/                   FSETP.GT.AND P1, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008093dc80 */
        /*14b0*/               @P1 BRA CC.NEU, 0x14e8;                                         /* 0x40000000c00005a7 */
        /*14b8*/                   MOV R10, R18;                                               /* 0x2800000048029de4 */
                                                                                               /* 0x228042b2b042b2b7 */
        /*14c8*/                   MOV R11, R19;                                               /* 0x280000004c02dde4 */
        /*14d0*/                   CAL 0x16e8;                                                 /* 0x5000000840011c07 */
        /*14d8*/                   MOV R30, R16;                                               /* 0x2800000040079de4 */
        /*14e0*/                   MOV R31, R17;                                               /* 0x280000004407dde4 */
        /*14e8*/                   MUFU.RCP64H.S R15, R29;                                     /* 0xc800000019d3dc10 */
        /*14f0*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*14f8*/                   SSY 0x1590;                                                 /* 0x6000000240001c07 */
                                                                                               /* 0x2202929292523297 */
        /*1508*/                   DFMA R10, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c29e01 */
        /*1510*/                   DFMA R12, R10, R10, R10;                                    /* 0x2014000028a31c01 */
        /*1518*/                   DMUL R10, R26, R30;                                         /* 0x5000000079a29c01 */
        /*1520*/                   DFMA R12, R14, R12, R14;                                    /* 0x201c000030e31c01 */
        /*1528*/                   DMUL R16, R10, R12;                                         /* 0x5000000030a41c01 */
        /*1530*/                   DFMA R14, -R28, R16, R10;                                   /* 0x2014000041c39e01 */
        /*1538*/                   DFMA R16, R12, R14, R16;                                    /* 0x2020000038c41c01 */
                                                                                               /* 0x22b2b042c2828047 */
        /*1548*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0xc], PT;                 /* 0x160f480030bfdc80 */
        /*1550*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*1558*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*1560*/                   FSETP.GT.AND P1, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008093dc80 */
        /*1568*/                   MOV R13, R29;                                               /* 0x2800000074035de4 */
        /*1570*/               @P1 BRA CC.NEU, 0x1580;                                         /* 0x40000000200005a7 */
        /*1578*/                   CAL 0x16e8;                                                 /* 0x50000005a0011c07 */
                                                                                               /* 0x22329280420042b7 */
        /*1588*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*1590*/                   LD.E.64 R12, [R2+0x4e218];                                  /* 0x8400138860231ca5 */
        /*1598*/                   MUFU.RCP64H R15, R29;                                       /* 0xc800000019d3dc00 */
        /*15a0*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*15a8*/                   SSY 0x1650;                                                 /* 0x6000000280001c07 */
        /*15b0*/                   DFMA R10, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c29e01 */
        /*15b8*/                   DFMA R18, R10, R10, R10;                                    /* 0x2014000028a49c01 */
                                                                                               /* 0x229233f202929257 */
        /*15c8*/                   DMUL R10, R24, R30;                                         /* 0x5000000079829c01 */
        /*15d0*/                   DFMA R20, R14, R18, R14;                                    /* 0x201c000048e51c01 */
        /*15d8*/                   DMUL R14, R20, R10;                                         /* 0x5000000029439c01 */
        /*15e0*/                   DFMA R18, -R28, R14, R10;                                   /* 0x2014000039c49e01 */
        /*15e8*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0xc], PT;                 /* 0x160f480030bfdc80 */
        /*15f0*/                   DADD R12, R16, R12;                                         /* 0x4800000031031c01 */
        /*15f8*/                   DFMA R16, R20, R18, R14;                                    /* 0x201c000049441c01 */
                                                                                               /* 0x22b2b042b2c28047 */
        /*1608*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*1610*/                   ST.E.64 [R2+0x4e218], R12;                                  /* 0x9400138860231ca5 */
        /*1618*/                   FSETP.GT.AND P1, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008093dc80 */
        /*1620*/               @P1 BRA CC.NEU, 0x1640;                                         /* 0x40000000600005a7 */
        /*1628*/                   MOV R12, R28;                                               /* 0x2800000070031de4 */
        /*1630*/                   MOV R13, R29;                                               /* 0x2800000074035de4 */
        /*1638*/                   CAL 0x16e8;                                                 /* 0x50000002a0011c07 */
                                                                                               /* 0x22c042c2b293f2b7 */
        /*1648*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*1650*/                   LD.E.64 R10, [R2+0x5dc18];                                  /* 0x8400177060229ca5 */
        /*1658*/                   DADD R16, R16, R10;                                         /* 0x4800000029041c01 */
        /*1660*/                   ST.E.64.S [R2+0x5dc18], R16;                                /* 0x9400177060241cb5 */
        /*1668*/                   IADD32I R4.CC, R4, 0x28;                                    /* 0x0c000000a0411c02 */
        /*1670*/                   IADD.X R5, RZ, R5;                                          /* 0x4800000017f15c43 */
        /*1678*/                   IADD32I R8.CC, R8, 0x5;                                     /* 0x0c00000014821c02 */
                                                                                               /* 0x2202c202c202c047 */
        /*1688*/                   IADD.X R38, RZ, R38;                                        /* 0x480000009bf99c43 */
        /*1690*/                   IADD32I R7.CC, R7, 0x5;                                     /* 0x0c0000001471dc02 */
        /*1698*/                   IADD.X R39, RZ, R39;                                        /* 0x480000009ff9dc43 */
        /*16a0*/                   IADD32I R6.CC, R6, 0x5;                                     /* 0x0c00000014619c02 */
        /*16a8*/                   IADD.X R40, RZ, R40;                                        /* 0x48000000a3fa1c43 */
        /*16b0*/                   IADD32I R0.CC, R0, 0x5;                                     /* 0x0c00000014001c02 */
        /*16b8*/                   IADD.X R41, RZ, R41;                                        /* 0x48000000a7fa5c43 */
                                                                                               /* 0x22c28042e2e042c7 */
        /*16c8*/                   IADD32I R32.CC, R32, 0x5;                                   /* 0x0c00000016081c02 */
        /*16d0*/                   IADD.X R42, RZ, R42;                                        /* 0x48000000abfa9c43 */
        /*16d8*/               @P0 BRA 0x140;                                                  /* 0x4003ffa9800001e7 */
        /*16e0*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*16e8*/                   LOP32I.AND R9, R13, 0x40000000;                             /* 0x3900000000d25c02 */
        /*16f0*/                   MOV32I R45, 0x1ff00000;                                     /* 0x187fc000000b5de2 */
        /*16f8*/                   ISETP.LT.U32.AND P1, PT, R9, c[0x2][0x24], PT;              /* 0x188e48009093dc03 */
                                                                                               /* 0x2292804200428047 */
        /*1708*/                   SEL R14, R45, c[0x2][0x28], !P1;                            /* 0x20124800a2d39c04 */
        /*1710*/                   LOP32I.AND R9, R11, 0x7f800000;                             /* 0x39fe000000b25c02 */
        /*1718*/                   MOV R15, R14;                                               /* 0x280000003803dde4 */
        /*1720*/                   ISETP.LT.U32.AND P1, PT, R9, c[0x2][0x2c], PT;              /* 0x188e4800b093dc03 */
        /*1728*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
        /*1730*/                   PBK 0x1a30;                                                 /* 0x6800000be0001c07 */
        /*1738*/                   DMUL R16, R12, R14;                                         /* 0x5000000038c41c01 */
                                                                                               /* 0x2282028042028047 */
        /*1748*/                   MUFU.RCP64H R20, R17;                                       /* 0xc800000019151c00 */
        /*1750*/                   SEL R34, R45, c[0x2][0x28], !P1;                            /* 0x20124800a2d89c04 */
        /*1758*/                   MOV R21, R20;                                               /* 0x2800000050055de4 */
        /*1760*/                   MOV32I R20, 0x1;                                            /* 0x1800000004051de2 */
        /*1768*/                   MOV R35, R34;                                               /* 0x280000008808dde4 */
        /*1770*/                   DFMA R18, -R16, R20, c[0x2][0x0];                           /* 0x2028880001049e01 */
        /*1778*/                   MOV R34, RZ;                                                /* 0x28000000fc089de4 */
                                                                                               /* 0x22d2929292925237 */
        /*1788*/                   DFMA R22, R18, R18, R18;                                    /* 0x2024000049259c01 */
        /*1790*/                   DMUL R18, R10, R34;                                         /* 0x5000000088a49c01 */
        /*1798*/                   DFMA R20, R20, R22, R20;                                    /* 0x2028000059451c01 */
        /*17a0*/                   DMUL R22, R18, R20;                                         /* 0x5000000051259c01 */
        /*17a8*/                   DFMA R16, -R16, R22, R18;                                   /* 0x2024000059041e01 */
        /*17b0*/                   DFMA R20, R20, R16, R22;                                    /* 0x202c000041451c01 */
        /*17b8*/                   DSETP.LEU.AND P1, PT, |R20|, RZ, PT;                        /* 0x1d8e0000fd43dc81 */
                                                                                               /* 0x2232804282c042b7 */
        /*17c8*/               @P1 BRA 0x19a8;                                                 /* 0x40000007600005e7 */
        /*17d0*/                   ISETP.GT.U32.AND P1, PT, R9, c[0x2][0x30], PT;              /* 0x1a0e4800c093dc03 */
        /*17d8*/                   DMUL R18, R14, R20;                                         /* 0x5000000050e49c01 */
        /*17e0*/                   SEL R16, R45, c[0x2][0x28], !P1;                            /* 0x20124800a2d41c04 */
        /*17e8*/                   MOV R17, R16;                                               /* 0x2800000040045de4 */
        /*17f0*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*17f8*/                   DMUL R20, R20, R16;                                         /* 0x5000000041451c01 */
                                                                                               /* 0x227202d292523257 */
        /*1808*/                   DMUL R18, R16, R18;                                         /* 0x5000000049049c01 */
        /*1810*/                   DMUL R20, R14, R20;                                         /* 0x5000000050e51c01 */
        /*1818*/                   DFMA R14, R12, R18, -R10;                                   /* 0x2014000048c39d01 */
        /*1820*/                   DFMA R16, R12, R20, -R10;                                   /* 0x2014000050c41d01 */
        /*1828*/                   DSETP.GT.AND P1, PT, |R14|, |R16|, PT;                      /* 0x1a0e000040e3dcc1 */
        /*1830*/                   SEL R9, R21, R19, P1;                                       /* 0x200200004d525c04 */
        /*1838*/                   SEL R16, R20, R18, P1;                                      /* 0x2002000049441c04 */
                                                                                               /* 0x2200420042b042c7 */
        /*1848*/                   FSETP.GTU.AND P1, PT, |R9|, 1.469367938527859385e-39, PT;   /* 0x260ec0040093dc80 */
        /*1850*/                   MOV R17, R9;                                                /* 0x2800000024045de4 */
        /*1858*/               @P1 BRK;                                                        /* 0xa8000000000005e7 */
        /*1860*/                   FSETP.LT.AND P1, PT, |R11|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000b3dc80 */
        /*1868*/                   LOP32I.AND R19, R16, 0xfffffffe;                            /* 0x3bfffffff904dc02 */
        /*1870*/                   MOV32I R14, 0x3ff00000;                                     /* 0x18ffc00000039de2 */
        /*1878*/                   MOV R17, R9;                                                /* 0x2800000024045de4 */
                                                                                               /* 0x22023280428042a7 */
        /*1888*/                   LOP32I.OR R18, R16, 0x1;                                    /* 0x3800000005049c42 */
        /*1890*/                   SEL R14, R14, c[0x2][0x34], !P1;                            /* 0x20124800d0e39c04 */
        /*1898*/                   MOV R16, R18;                                               /* 0x2800000048041de4 */
        /*18a0*/                   MOV R15, R14;                                               /* 0x280000003803dde4 */
        /*18a8*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
        /*18b0*/                   DMUL R12, R12, R14;                                         /* 0x5000000038c31c01 */
        /*18b8*/                   DMUL R10, R10, R14;                                         /* 0x5000000038a29c01 */
                                                                                               /* 0x220282d292327207 */
        /*18c8*/                   MOV R14, R19;                                               /* 0x280000004c039de4 */
        /*18d0*/                   MOV R15, R9;                                                /* 0x280000002403dde4 */
        /*18d8*/                   DFMA R16, R16, R12, -R10;                                   /* 0x2014000031041d01 */
        /*18e0*/                   DFMA R14, R14, R12, -R10;                                   /* 0x2014000030e39d01 */
        /*18e8*/                   DSETP.GT.AND P1, PT, |R14|, |R16|, PT;                      /* 0x1a0e000040e3dcc1 */
        /*18f0*/                   SEL R14, R18, R19, P1;                                      /* 0x200200004d239c04 */
        /*18f8*/                   IADD32I R19.CC, R14, 0x1;                                   /* 0x0c00000004e4dc02 */
                                                                                               /* 0x22020042c20042b7 */
        /*1908*/                   LOP32I.AND R15, R14, 0x1;                                   /* 0x3800000004e3dc02 */
        /*1910*/                   IADD.X R18, RZ, R9;                                         /* 0x4800000027f49c43 */
        /*1918*/                   IADD32I R16.CC, R14, -0x1;                                  /* 0x0ffffffffce41c02 */
        /*1920*/                   ISETP.EQ.U32.AND P1, PT, R15, 0x1, PT;                      /* 0x190ec00004f3dc03 */
        /*1928*/                   IADD32I.X R15, R9, -0x1;                                    /* 0x0bfffffffc93dc42 */
        /*1930*/                   SEL R18, R9, R18, !P1;                                      /* 0x2012000048949c04 */
        /*1938*/                   SEL R19, R14, R19, !P1;                                     /* 0x201200004ce4dc04 */
                                                                                               /* 0x2292621262020267 */
        /*1948*/                   SEL R14, R16, R14, !P1;                                     /* 0x2012000039039c04 */
        /*1950*/                   SEL R9, R15, R9, !P1;                                       /* 0x2012000024f25c04 */
        /*1958*/                   MOV R16, R19;                                               /* 0x280000004c041de4 */
        /*1960*/                   MOV R17, R18;                                               /* 0x2800000048045de4 */
        /*1968*/                   MOV R15, R9;                                                /* 0x280000002403dde4 */
        /*1970*/                   DFMA R16, R12, R16, -R10;                                   /* 0x2014000040c41d01 */
        /*1978*/                   DFMA R10, R12, R14, -R10;                                   /* 0x2014000038c29d01 */
                                                                                               /* 0x2282b2d2e04202d7 */
        /*1988*/                   DSETP.GT.AND P1, PT, |R16|, |R10|, PT;                      /* 0x1a0e00002903dcc1 */
        /*1990*/                   SEL R16, R14, R19, P1;                                      /* 0x200200004ce41c04 */
        /*1998*/                   SEL R17, R9, R18, P1;                                       /* 0x2002000048945c04 */
        /*19a0*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*19a8*/                   DSETP.EQ.AND P1, PT, R20, RZ, PT;                           /* 0x190e0000fd43dc01 */
        /*19b0*/               @P1 BRA 0x1a20;                                                 /* 0x40000001a00005e7 */
        /*19b8*/                   MUFU.RCP64H R14, R13;                                       /* 0xc800000018d39c00 */
                                                                                               /* 0x22027202d2d28047 */
        /*19c8*/                   MOV R15, R14;                                               /* 0x280000003803dde4 */
        /*19d0*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
        /*19d8*/                   DSETP.GT.AND P2, PT, |R14|, RZ, PT;                         /* 0x1a0e0000fce5dc81 */
        /*19e0*/              @!P2 DSETP.NEU.AND P1, PT, |R12|, +INF , PT;                     /* 0x1e8edffc00c3e881 */
        /*19e8*/              @!P2 SEL R9, R12, R14, P1;                                       /* 0x2002000038c26804 */
        /*19f0*/              @!P2 SEL R12, R13, R15, P1;                                      /* 0x200200003cd32804 */
        /*19f8*/              @!P2 MOV R14, R9;                                                /* 0x280000002403a9e4 */
                                                                                               /* 0x2202e2b202e20287 */
        /*1a08*/              @!P2 MOV R15, R12;                                               /* 0x280000003003e9e4 */
        /*1a10*/                   DMUL R16, R10, R14;                                         /* 0x5000000038a41c01 */
        /*1a18*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1a20*/                   DMUL R16, R10, R12;                                         /* 0x5000000030a41c01 */
        /*1a28*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1a30*/                   RET;                                                        /* 0x9000000000001de7 */
        /*1a38*/                   ISETP.LT.U32.AND P1, PT, R15, c[0x2][0x38], PT;             /* 0x188e4800e0f3dc03 */
                                                                                               /* 0x229282c292b2b047 */
        /*1a48*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*1a50*/                   PBK 0x1b98;                                                 /* 0x6800000500001c07 */
        /*1a58*/               @P1 BRA 0x1aa8;                                                 /* 0x40000001200005e7 */
        /*1a60*/                   DFMA.RM R12, R12, R14, R16;                                 /* 0x20a0000038c31c01 */
        /*1a68*/                   IADD32I R14.CC, R12, 0x1;                                   /* 0x0c00000004c39c02 */
        /*1a70*/                   IADD.X R15, RZ, R13;                                        /* 0x4800000037f3dc43 */
        /*1a78*/                   DFMA.RP R10, -R12, R14, R10;                                /* 0x2114000038c29e01 */
                                                                                               /* 0x22c2b2d2e04202d7 */
        /*1a88*/                   DSETP.GT.AND P1, PT, R10, RZ, PT;                           /* 0x1a0e0000fca3dc01 */
        /*1a90*/                   SEL R12, R14, R12, P1;                                      /* 0x2002000030e31c04 */
        /*1a98*/                   SEL R13, R15, R13, P1;                                      /* 0x2002000034f35c04 */
        /*1aa0*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1aa8*/                   DSETP.EQU.AND P1, PT, R10, RZ, PT;                          /* 0x1d0e0000fca3dc01 */
        /*1ab0*/               @P1 BRA 0x1b80;                                                 /* 0x40000003200005e7 */
        /*1ab8*/                   ISETP.LT.AND P1, PT, R11, RZ, PT;                           /* 0x188e0000fcb3dc23 */
                                                                                               /* 0x228202b2c2b20047 */
        /*1ac8*/               @P1 MOV32I R12, 0x0;                                            /* 0x18000000000305e2 */
        /*1ad0*/               @P1 MOV32I R13, 0xfff80000;                                     /* 0x1bffe000000345e2 */
        /*1ad8*/               @P1 BRK;                                                        /* 0xa8000000000005e7 */
        /*1ae0*/                   ISETP.GT.AND P1, PT, R11, c[0x2][0x3c], PT;                 /* 0x1a0e4800f0b3dc23 */
        /*1ae8*/               @P1 BRA 0x1b80;                                                 /* 0x40000002400005e7 */
        /*1af0*/                   DMUL R10, R10, 8.11296384146066816958e+31;                  /* 0x5000d1a400a29c01 */
        /*1af8*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
                                                                                               /* 0x2292329282028047 */
        /*1b08*/                   MUFU.RSQ64H R15, R11;                                       /* 0xc80000001cb3dc00 */
        /*1b10*/                   MOV32I R12, 0x0;                                            /* 0x1800000000031de2 */
        /*1b18*/                   DMUL R16, R14, R14;                                         /* 0x5000000038e41c01 */
        /*1b20*/                   MOV32I R13, 0x3fe00000;                                     /* 0x18ff800000035de2 */
        /*1b28*/                   DFMA R16, -R10, R16, c[0x2][0x0];                           /* 0x2020880000a41e01 */
        /*1b30*/                   DFMA R12, R16, 0.375, R12;                                  /* 0x2018cff601031c01 */
        /*1b38*/                   DMUL R16, R14, R16;                                         /* 0x5000000040e41c01 */
                                                                                               /* 0x22e0429292820297 */
        /*1b48*/                   DFMA R14, R12, R16, R14;                                    /* 0x201c000040c39c01 */
        /*1b50*/                   DMUL R12, R10, R14;                                         /* 0x5000000038a31c01 */
        /*1b58*/                   IADD32I R15, R15, -0x100000;                                /* 0x0bffc00000f3dc02 */
        /*1b60*/                   DFMA R10, -R12, R12, R10;                                   /* 0x2014000030c29e01 */
        /*1b68*/                   DFMA R12, R14, R10, R12;                                    /* 0x2018000028e31c01 */
        /*1b70*/                   IADD32I R13, R13, -0x3500000;                               /* 0x0bf2c00000d35c02 */
        /*1b78*/                   BRK;                                                        /* 0xa800000000001de7 */
                                                                                               /* 0x200002e20042b207 */
        /*1b88*/                   DADD R12, R10, R10;                                         /* 0x4800000028a31c01 */
        /*1b90*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1b98*/                   MOV R28, R12;                                               /* 0x2800000030071de4 */
        /*1ba0*/                   MOV R29, R13;                                               /* 0x2800000034075de4 */
        /*1ba8*/                   RET;                                                        /* 0x9000000000001de7 */
        /*1bb0*/                   BRA 0x1bb0;                                                 /* 0x4003ffffe0001de7 */
        /*1bb8*/                   NOP;                                                        /* 0x4000000000001de4 */
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
                                                                                              /* 0x23f240423042b047 */
        /*0088*/                   IADD.X R3, R0, R3;                                         /* 0x480000000c00dc43 */
        /*0090*/                   MOV32I R0, 0xfffffc18;                                     /* 0x1bfffff060001de2 */
        /*0098*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*00a0*/                   IADD32I R0, R0, 0x5;                                       /* 0x0800000014001c02 */
        /*00a8*/                   LD.E.64 R4, [R2+0x4e210];                                  /* 0x8400138840211ca5 */
        /*00b0*/                   MOV32I R8, 0x1;                                            /* 0x1800000004021de2 */
        /*00b8*/                   ISETP.NE.AND P0, PT, R0, RZ, PT;                           /* 0x1a8e0000fc01dc23 */
                                                                                              /* 0x2292929252329287 */
        /*00c8*/                   MUFU.RCP64H R9, R7;                                        /* 0xc800000018725c00 */
        /*00d0*/                   DFMA R10, -R6, R8, c[0x2][0x0];                            /* 0x2010880000629e01 */
        /*00d8*/                   DFMA R10, R10, R10, R10;                                   /* 0x2014000028a29c01 */
        /*00e0*/                   DMUL R4, R4, c[0x0][0x148];                                /* 0x5000400520411c01 */
        /*00e8*/                   DFMA R8, R8, R10, R8;                                      /* 0x2010000028821c01 */
        /*00f0*/                   DMUL R10, R4, R8;                                          /* 0x5000000020429c01 */
        /*00f8*/                   DFMA R12, -R6, R10, R4;                                    /* 0x2008000028631e01 */
                                                                                              /* 0x2232b2b2c2828207 */
        /*0108*/                   DFMA R8, R8, R12, R10;                                     /* 0x2014000030821c01 */
        /*0110*/                   FSET.GTU.AND RZ.CC, |R5|, c[0x2][0x8], PT;                 /* 0x160f4800205fdc80 */
        /*0118*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*0120*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*0128*/               @P1 BRA CC.NEU, 0x138;                                         /* 0x40000000200005a7 */
        /*0130*/                   CAL 0x9f0;                                                 /* 0x50000022e0011c07 */
        /*0138*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
                                                                                              /* 0x2252329283f23047 */
        /*0148*/                   LD.E.64 R10, [R2+0x5dc10];                                 /* 0x8400177040229ca5 */
        /*0150*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*0158*/                   LD.E.64 R4, [R2+0x2ee10];                                  /* 0x84000bb840211ca5 */
        /*0160*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*0168*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*0170*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
        /*0178*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
                                                                                              /* 0x2200429232929297 */
        /*0188*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*0190*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0198*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
        /*01a0*/                   DFMA R18, R12, R14, R16;                                   /* 0x2020000038c49c01 */
        /*01a8*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
        /*01b0*/                   FFMA R8, RZ, R7, R19;                                      /* 0x302600001ff21c00 */
        /*01b8*/                   ST.E.64 [R2+0x2ee10], R4;                                  /* 0x94000bb840211ca5 */
                                                                                              /* 0x2202b2b042b2c277 */
        /*01c8*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*01d0*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003083dc80 */
        /*01d8*/               @P1 BRA CC.NEU, 0x210;                                         /* 0x40000000c00005a7 */
        /*01e0*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*01e8*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*01f0*/                   CAL 0x9f0;                                                 /* 0x5000001fe0011c07 */
        /*01f8*/                   MOV R18, R8;                                               /* 0x2800000020049de4 */
                                                                                              /* 0x2203f232323232b7 */
        /*0208*/                   MOV R19, R9;                                               /* 0x280000002404dde4 */
        /*0210*/                   LD.E.64 R12, [R2+0x3e810];                                 /* 0x84000fa040231ca5 */
        /*0218*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*0220*/                   LD.E.64 R16, [R2+0x4e210];                                 /* 0x8400138840241ca5 */
        /*0228*/                   LD.E.64 R4, [R2+0x2ee10];                                  /* 0x84000bb840211ca5 */
        /*0230*/                   LD.E.64 R8, [R2+0xfa10];                                   /* 0x840003e840221ca5 */
        /*0238*/                   DADD R14, R18, R12;                                        /* 0x4800000031239c01 */
                                                                                              /* 0x22329042e042e047 */
        /*0248*/                   LD.E.64 R12, [R2+0x1f410];                                 /* 0x840007d040231ca5 */
        /*0250*/                   MOV32I R18, 0x1;                                           /* 0x1800000004049de2 */
        /*0258*/                   ST.E.64 [R2+0x3e810], R14;                                 /* 0x94000fa040239ca5 */
        /*0260*/                   MUFU.RCP64H R19, R7;                                       /* 0xc80000001874dc00 */
        /*0268*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
        /*0270*/                   DFMA R20, -R6, R18, c[0x2][0x0];                           /* 0x2024880000651e01 */
        /*0278*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
                                                                                              /* 0x2202029252329257 */
        /*0288*/                   DMUL R16, R16, c[0x0][0x148];                              /* 0x5000400521041c01 */
        /*0290*/                   DFMA R18, R18, R20, R18;                                   /* 0x2024000051249c01 */
        /*0298*/                   DMUL R20, R16, R18;                                        /* 0x5000000049051c01 */
        /*02a0*/                   DFMA R4, R4, c[0x0][0x148], R8;                            /* 0x2010400520411c01 */
        /*02a8*/                   DFMA R8, -R6, R20, R16;                                    /* 0x2020000050621e01 */
        /*02b0*/                   DFMA R8, R18, R8, R20;                                     /* 0x2028000021221c01 */
        /*02b8*/                   ST.E.64 [R2+0xfa10], R4;                                   /* 0x940003e840211ca5 */
                                                                                              /* 0x2202b2c0428203f7 */
        /*02c8*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                /* 0x160f4800211fdc80 */
        /*02d0*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*02d8*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*02e0*/                   ST.E.64 [R2+0x1f410], R12;                                 /* 0x940007d040231ca5 */
        /*02e8*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*02f0*/               @P1 BRA CC.NEU, 0x318;                                         /* 0x40000000800005a7 */
        /*02f8*/                   MOV R4, R16;                                               /* 0x2800000040011de4 */
                                                                                              /* 0x2283f2323042b2b7 */
        /*0308*/                   MOV R5, R17;                                               /* 0x2800000044015de4 */
        /*0310*/                   CAL 0x9f0;                                                 /* 0x5000001b60011c07 */
        /*0318*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*0320*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*0328*/                   LD.E.64 R10, [R2+0x5dc10];                                 /* 0x8400177040229ca5 */
        /*0330*/                   LD.E.64 R4, [R2+0x2ee10];                                  /* 0x84000bb840211ca5 */
        /*0338*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
                                                                                              /* 0x2232929292523297 */
        /*0348*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*0350*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
        /*0358*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*0360*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*0368*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0370*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
        /*0378*/                   DFMA R18, R12, R14, R16;                                   /* 0x2020000038c49c01 */
                                                                                              /* 0x2202b2c272004297 */
        /*0388*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
        /*0390*/                   FFMA R8, RZ, R7, R19;                                      /* 0x302600001ff21c00 */
        /*0398*/                   ST.E.64 [R2+0x2ee10], R4;                                  /* 0x94000bb840211ca5 */
        /*03a0*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*03a8*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003083dc80 */
        /*03b0*/               @P1 BRA CC.NEU, 0x3e8;                                         /* 0x40000000c00005a7 */
        /*03b8*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
                                                                                              /* 0x22323232b042b2b7 */
        /*03c8*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*03d0*/                   CAL 0x9f0;                                                 /* 0x5000001860011c07 */
        /*03d8*/                   MOV R18, R8;                                               /* 0x2800000020049de4 */
        /*03e0*/                   MOV R19, R9;                                               /* 0x280000002404dde4 */
        /*03e8*/                   LD.E.64 R12, [R2+0x3e810];                                 /* 0x84000fa040231ca5 */
        /*03f0*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*03f8*/                   LD.E.64 R16, [R2+0x4e210];                                 /* 0x8400138840241ca5 */
                                                                                              /* 0x22e042e04203f237 */
        /*0408*/                   LD.E.64 R4, [R2+0x2ee10];                                  /* 0x84000bb840211ca5 */
        /*0410*/                   LD.E.64 R8, [R2+0xfa10];                                   /* 0x840003e840221ca5 */
        /*0418*/                   DADD R14, R18, R12;                                        /* 0x4800000031239c01 */
        /*0420*/                   LD.E.64 R12, [R2+0x1f410];                                 /* 0x840007d040231ca5 */
        /*0428*/                   MOV32I R18, 0x1;                                           /* 0x1800000004049de2 */
        /*0430*/                   ST.E.64 [R2+0x3e810], R14;                                 /* 0x94000fa040239ca5 */
        /*0438*/                   MUFU.RCP64H R19, R7;                                       /* 0xc80000001874dc00 */
                                                                                              /* 0x2252329252329047 */
        /*0448*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
        /*0450*/                   DFMA R20, -R6, R18, c[0x2][0x0];                           /* 0x2024880000651e01 */
        /*0458*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*0460*/                   DMUL R16, R16, c[0x0][0x148];                              /* 0x5000400521041c01 */
        /*0468*/                   DFMA R18, R18, R20, R18;                                   /* 0x2024000051249c01 */
        /*0470*/                   DMUL R20, R16, R18;                                        /* 0x5000000049051c01 */
        /*0478*/                   DFMA R4, R4, c[0x0][0x148], R8;                            /* 0x2010400520411c01 */
                                                                                              /* 0x22028203f0420297 */
        /*0488*/                   DFMA R8, -R6, R20, R16;                                    /* 0x2020000050621e01 */
        /*0490*/                   DFMA R8, R18, R8, R20;                                     /* 0x2028000021221c01 */
        /*0498*/                   ST.E.64 [R2+0xfa10], R4;                                   /* 0x940003e840211ca5 */
        /*04a0*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                /* 0x160f4800211fdc80 */
        /*04a8*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*04b0*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*04b8*/                   ST.E.64 [R2+0x1f410], R12;                                 /* 0x940007d040231ca5 */
                                                                                              /* 0x223042b2b042b2c7 */
        /*04c8*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*04d0*/               @P1 BRA CC.NEU, 0x4f0;                                         /* 0x40000000600005a7 */
        /*04d8*/                   MOV R4, R16;                                               /* 0x2800000040011de4 */
        /*04e0*/                   MOV R5, R17;                                               /* 0x2800000044015de4 */
        /*04e8*/                   CAL 0x9f0;                                                 /* 0x5000001400011c07 */
        /*04f0*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*04f8*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
                                                                                              /* 0x229252329283f237 */
        /*0508*/                   LD.E.64 R10, [R2+0x5dc10];                                 /* 0x8400177040229ca5 */
        /*0510*/                   LD.E.64 R4, [R2+0x2ee10];                                  /* 0x84000bb840211ca5 */
        /*0518*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*0520*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*0528*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
        /*0530*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*0538*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
                                                                                              /* 0x2272004292329297 */
        /*0548*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0550*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
        /*0558*/                   DFMA R18, R12, R14, R16;                                   /* 0x2020000038c49c01 */
        /*0560*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
        /*0568*/                   FFMA R8, RZ, R7, R19;                                      /* 0x302600001ff21c00 */
        /*0570*/                   ST.E.64 [R2+0x2ee10], R4;                                  /* 0x94000bb840211ca5 */
        /*0578*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
                                                                                              /* 0x22b042b2b042b2c7 */
        /*0588*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003083dc80 */
        /*0590*/               @P1 BRA CC.NEU, 0x5c0;                                         /* 0x40000000a00005a7 */
        /*0598*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*05a0*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*05a8*/                   CAL 0x9f0;                                                 /* 0x5000001100011c07 */
        /*05b0*/                   MOV R18, R8;                                               /* 0x2800000020049de4 */
        /*05b8*/                   MOV R19, R9;                                               /* 0x280000002404dde4 */
                                                                                              /* 0x22e203f232323237 */
        /*05c8*/                   LD.E.64 R12, [R2+0x3e810];                                 /* 0x84000fa040231ca5 */
        /*05d0*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*05d8*/                   LD.E.64 R16, [R2+0x4e210];                                 /* 0x8400138840241ca5 */
        /*05e0*/                   LD.E.64 R4, [R2+0x2ee10];                                  /* 0x84000bb840211ca5 */
        /*05e8*/                   LD.E.64 R8, [R2+0xfa10];                                   /* 0x840003e840221ca5 */
        /*05f0*/                   DADD R14, R18, R12;                                        /* 0x4800000031239c01 */
        /*05f8*/                   LD.E.64 R12, [R2+0x1f410];                                 /* 0x840007d040231ca5 */
                                                                                              /* 0x229280423042e047 */
        /*0608*/                   ST.E.64 [R2+0x3e810], R14;                                 /* 0x94000fa040239ca5 */
        /*0610*/                   MUFU.RCP64H R19, R7;                                       /* 0xc80000001874dc00 */
        /*0618*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
        /*0620*/                   DMUL R16, R16, c[0x0][0x148];                              /* 0x5000400521041c01 */
        /*0628*/                   MOV32I R18, 0x1;                                           /* 0x1800000004049de2 */
        /*0630*/                   DFMA R4, R4, c[0x0][0x148], R8;                            /* 0x2010400520411c01 */
        /*0638*/                   DFMA R20, -R6, R18, c[0x2][0x0];                           /* 0x2024880000651e01 */
                                                                                              /* 0x23f0420292929297 */
        /*0648*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*0650*/                   DFMA R18, R18, R20, R18;                                   /* 0x2024000051249c01 */
        /*0658*/                   DMUL R20, R16, R18;                                        /* 0x5000000049051c01 */
        /*0660*/                   DFMA R8, -R6, R20, R16;                                    /* 0x2020000050621e01 */
        /*0668*/                   DFMA R8, R18, R8, R20;                                     /* 0x2028000021221c01 */
        /*0670*/                   ST.E.64 [R2+0xfa10], R4;                                   /* 0x940003e840211ca5 */
        /*0678*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                /* 0x160f4800211fdc80 */
                                                                                              /* 0x22b042b2c0428207 */
        /*0688*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*0690*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*0698*/                   ST.E.64 [R2+0x1f410], R12;                                 /* 0x940007d040231ca5 */
        /*06a0*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*06a8*/               @P1 BRA CC.NEU, 0x6d0;                                         /* 0x40000000800005a7 */
        /*06b0*/                   MOV R4, R16;                                               /* 0x2800000040011de4 */
        /*06b8*/                   MOV R5, R17;                                               /* 0x2800000044015de4 */
                                                                                              /* 0x229283f2323042b7 */
        /*06c8*/                   CAL 0x9f0;                                                 /* 0x5000000c80011c07 */
        /*06d0*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*06d8*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*06e0*/                   LD.E.64 R10, [R2+0x5dc10];                                 /* 0x8400177040229ca5 */
        /*06e8*/                   LD.E.64 R4, [R2+0x2ee10];                                  /* 0x84000bb840211ca5 */
        /*06f0*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*06f8*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
                                                                                              /* 0x2292329292925237 */
        /*0708*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
        /*0710*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*0718*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*0720*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0728*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
        /*0730*/                   DFMA R18, R12, R14, R16;                                   /* 0x2020000038c49c01 */
        /*0738*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
                                                                                              /* 0x22b042b2c2720047 */
        /*0748*/                   FFMA R8, RZ, R7, R19;                                      /* 0x302600001ff21c00 */
        /*0750*/                   ST.E.64 [R2+0x2ee10], R4;                                  /* 0x94000bb840211ca5 */
        /*0758*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*0760*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003083dc80 */
        /*0768*/               @P1 BRA CC.NEU, 0x7a0;                                         /* 0x40000000c00005a7 */
        /*0770*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*0778*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
                                                                                              /* 0x2232323232b042b7 */
        /*0788*/                   CAL 0x9f0;                                                 /* 0x5000000980011c07 */
        /*0790*/                   MOV R18, R8;                                               /* 0x2800000020049de4 */
        /*0798*/                   MOV R19, R9;                                               /* 0x280000002404dde4 */
        /*07a0*/                   LD.E.64 R12, [R2+0x3e810];                                 /* 0x84000fa040231ca5 */
        /*07a8*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*07b0*/                   LD.E.64 R16, [R2+0x4e210];                                 /* 0x8400138840241ca5 */
        /*07b8*/                   LD.E.64 R4, [R2+0x2ee10];                                  /* 0x84000bb840211ca5 */
                                                                                              /* 0x2202e042e04203f7 */
        /*07c8*/                   LD.E.64 R8, [R2+0xfa10];                                   /* 0x840003e840221ca5 */
        /*07d0*/                   DADD R14, R18, R12;                                        /* 0x4800000031239c01 */
        /*07d8*/                   LD.E.64 R12, [R2+0x1f410];                                 /* 0x840007d040231ca5 */
        /*07e0*/                   MOV32I R18, 0x1;                                           /* 0x1800000004049de2 */
        /*07e8*/                   ST.E.64 [R2+0x3e810], R14;                                 /* 0x94000fa040239ca5 */
        /*07f0*/                   MUFU.RCP64H R19, R7;                                       /* 0xc80000001874dc00 */
        /*07f8*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
                                                                                              /* 0x2292523292523297 */
        /*0808*/                   DFMA R20, -R6, R18, c[0x2][0x0];                           /* 0x2024880000651e01 */
        /*0810*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*0818*/                   DMUL R16, R16, c[0x0][0x148];                              /* 0x5000400521041c01 */
        /*0820*/                   DFMA R18, R18, R20, R18;                                   /* 0x2024000051249c01 */
        /*0828*/                   DMUL R20, R16, R18;                                        /* 0x5000000049051c01 */
        /*0830*/                   DFMA R4, R4, c[0x0][0x148], R8;                            /* 0x2010400520411c01 */
        /*0838*/                   DFMA R8, -R6, R20, R16;                                    /* 0x2020000050621e01 */
                                                                                              /* 0x22c0428203f04207 */
        /*0848*/                   DFMA R8, R18, R8, R20;                                     /* 0x2028000021221c01 */
        /*0850*/                   ST.E.64 [R2+0xfa10], R4;                                   /* 0x940003e840211ca5 */
        /*0858*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                /* 0x160f4800211fdc80 */
        /*0860*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*0868*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*0870*/                   ST.E.64 [R2+0x1f410], R12;                                 /* 0x940007d040231ca5 */
        /*0878*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
                                                                                              /* 0x22323042b2b042b7 */
        /*0888*/               @P1 BRA CC.NEU, 0x8a8;                                         /* 0x40000000600005a7 */
        /*0890*/                   MOV R4, R16;                                               /* 0x2800000040011de4 */
        /*0898*/                   MOV R5, R17;                                               /* 0x2800000044015de4 */
        /*08a0*/                   CAL 0x9f0;                                                 /* 0x5000000520011c07 */
        /*08a8*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*08b0*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*08b8*/                   LD.E.64 R10, [R2+0x5dc10];                                 /* 0x8400177040229ca5 */
                                                                                              /* 0x22929252329283f7 */
        /*08c8*/                   LD.E.64 R4, [R2+0x2ee10];                                  /* 0x84000bb840211ca5 */
        /*08d0*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*08d8*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*08e0*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
        /*08e8*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*08f0*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*08f8*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
                                                                                              /* 0x22c2720042923297 */
        /*0908*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
        /*0910*/                   DFMA R16, R12, R14, R16;                                   /* 0x2020000038c41c01 */
        /*0918*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
        /*0920*/                   FFMA R8, RZ, R7, R17;                                      /* 0x302200001ff21c00 */
        /*0928*/                   ST.E.64 [R2+0x2ee10], R4;                                  /* 0x94000bb840211ca5 */
        /*0930*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*0938*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003083dc80 */
                                                                                              /* 0x2232b042b2b042b7 */
        /*0948*/               @P1 BRA CC.NEU, 0x978;                                         /* 0x40000000a00005a7 */
        /*0950*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*0958*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*0960*/                   CAL 0x9f0;                                                 /* 0x5000000220011c07 */
        /*0968*/                   MOV R16, R8;                                               /* 0x2800000020041de4 */
        /*0970*/                   MOV R17, R9;                                               /* 0x2800000024045de4 */
        /*0978*/                   LD.E.64 R4, [R2+0x3e810];                                  /* 0x84000fa040211ca5 */
                                                                                              /* 0x22e2e04293f23237 */
        /*0988*/                   LD.E.64 R6, [R2+0x2ee10];                                  /* 0x84000bb840219ca5 */
        /*0990*/                   LD.E.64 R8, [R2+0xfa10];                                   /* 0x840003e840221ca5 */
        /*0998*/                   LD.E.64 R12, [R2+0x1f410];                                 /* 0x840007d040231ca5 */
        /*09a0*/                   DADD R4, R16, R4;                                          /* 0x4800000011011c01 */
        /*09a8*/                   ST.E.64 [R2+0x3e810], R4;                                  /* 0x94000fa040211ca5 */
        /*09b0*/                   DFMA R6, R6, c[0x0][0x148], R8;                            /* 0x2010400520619c01 */
        /*09b8*/                   LD.E.64 R10, [R2+0x3e810];                                 /* 0x84000fa040229ca5 */
                                                                                              /* 0x228042e2e04293f7 */
        /*09c8*/                   ST.E.64 [R2+0xfa10], R6;                                   /* 0x940003e840219ca5 */
        /*09d0*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*09d8*/                   ST.E.64 [R2+0x1f410], R12;                                 /* 0x940007d040231ca5 */
        /*09e0*/               @P0 BRA 0x98;                                                  /* 0x4003ffdac00001e7 */
        /*09e8*/                   EXIT;                                                      /* 0x8000000000001de7 */
        /*09f0*/                   LOP32I.AND R8, R7, 0x40000000;                             /* 0x3900000000721c02 */
        /*09f8*/                   MOV32I R12, 0x1ff00000;                                    /* 0x187fc00000031de2 */
                                                                                              /* 0x22820282b0420047 */
        /*0a08*/                   ISETP.LT.U32.AND P1, PT, R8, c[0x2][0x10], PT;             /* 0x188e48004083dc03 */
        /*0a10*/                   LOP32I.AND R13, R5, 0x7f800000;                            /* 0x39fe000000535c02 */
        /*0a18*/                   MOV R8, RZ;                                                /* 0x28000000fc021de4 */
        /*0a20*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0a28*/                   SEL R9, R12, c[0x2][0x14], !P1;                            /* 0x2012480050c25c04 */
        /*0a30*/                   DMUL R10, R6, R8;                                          /* 0x5000000020629c01 */
        /*0a38*/                   ISETP.LT.U32.AND P1, PT, R13, c[0x2][0x18], PT;            /* 0x188e480060d3dc03 */
                                                                                              /* 0x2292523282028047 */
        /*0a48*/                   MUFU.RCP64H R15, R11;                                      /* 0xc800000018b3dc00 */
        /*0a50*/                   MOV R16, RZ;                                               /* 0x28000000fc041de4 */
        /*0a58*/                   DFMA R18, -R10, R14, c[0x2][0x0];                          /* 0x201c880000a49e01 */
        /*0a60*/                   SEL R17, R12, c[0x2][0x14], !P1;                           /* 0x2012480050c45c04 */
        /*0a68*/                   DFMA R18, R18, R18, R18;                                   /* 0x2024000049249c01 */
        /*0a70*/                   DMUL R16, R4, R16;                                         /* 0x5000000040441c01 */
        /*0a78*/                   DFMA R14, R14, R18, R14;                                   /* 0x201c000048e39c01 */
                                                                                              /* 0x22c042b2d2929297 */
        /*0a88*/                   DMUL R20, R16, R14;                                        /* 0x5000000039051c01 */
        /*0a90*/                   DFMA R10, -R10, R20, R16;                                  /* 0x2020000050a29e01 */
        /*0a98*/                   DFMA R14, R14, R10, R20;                                   /* 0x2028000028e39c01 */
        /*0aa0*/                   DSETP.LEU.AND P1, PT, |R14|, RZ, PT;                       /* 0x1d8e0000fce3dc81 */
        /*0aa8*/               @P1 BRA 0xc60;                                                 /* 0x40000006c00005e7 */
        /*0ab0*/                   ISETP.GT.U32.AND P1, PT, R13, c[0x2][0x1c], PT;            /* 0x1a0e480070d3dc03 */
        /*0ab8*/                   MOV R10, RZ;                                               /* 0x28000000fc029de4 */
                                                                                              /* 0x2292523252327207 */
        /*0ac8*/                   SEL R11, R12, c[0x2][0x14], !P1;                           /* 0x2012480050c2dc04 */
        /*0ad0*/                   DMUL R12, R8, R14;                                         /* 0x5000000038831c01 */
        /*0ad8*/                   DMUL R14, R14, R10;                                        /* 0x5000000028e39c01 */
        /*0ae0*/                   DMUL R10, R10, R12;                                        /* 0x5000000030a29c01 */
        /*0ae8*/                   DMUL R14, R8, R14;                                         /* 0x5000000038839c01 */
        /*0af0*/                   DFMA R8, R6, R10, -R4;                                     /* 0x2008000028621d01 */
        /*0af8*/                   DFMA R12, R6, R14, -R4;                                    /* 0x2008000038631d01 */
                                                                                              /* 0x220042e2c27202d7 */
        /*0b08*/                   DSETP.GT.AND P1, PT, |R8|, |R12|, PT;                      /* 0x1a0e00003083dcc1 */
        /*0b10*/                   SEL R9, R15, R11, P1;                                      /* 0x200200002cf25c04 */
        /*0b18*/                   SEL R8, R14, R10, P1;                                      /* 0x2002000028e21c04 */
        /*0b20*/                   FSETP.GTU.AND P1, PT, |R9|, 1.469367938527859385e-39, PT;  /* 0x260ec0040093dc80 */
        /*0b28*/               @P1 RET;                                                       /* 0x90000000000005e7 */
        /*0b30*/                   FSETP.LT.AND P1, PT, |R5|, 1.5046327690525280102e-36, PT;  /* 0x208ec1000053dc80 */
        /*0b38*/                   LOP32I.AND R14, R8, 0xfffffffe;                            /* 0x3bfffffff8839c02 */
                                                                                              /* 0x2202328230427047 */
        /*0b48*/                   MOV32I R11, 0x3ff00000;                                    /* 0x18ffc0000002dde2 */
        /*0b50*/                   MOV R10, RZ;                                               /* 0x28000000fc029de4 */
        /*0b58*/                   LOP32I.OR R15, R8, 0x1;                                    /* 0x380000000483dc42 */
        /*0b60*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0b68*/                   SEL R11, R11, c[0x2][0x20], !P1;                           /* 0x2012480080b2dc04 */
        /*0b70*/                   DMUL R6, R6, R10;                                          /* 0x5000000028619c01 */
        /*0b78*/                   DMUL R4, R4, R10;                                          /* 0x5000000028411c01 */
                                                                                              /* 0x220282d292328047 */
        /*0b88*/                   MOV R10, R15;                                              /* 0x280000003c029de4 */
        /*0b90*/                   MOV R11, R9;                                               /* 0x280000002402dde4 */
        /*0b98*/                   DFMA R12, R8, R6, -R4;                                     /* 0x2008000018831d01 */
        /*0ba0*/                   DFMA R10, R10, R6, -R4;                                    /* 0x2008000018a29d01 */
        /*0ba8*/                   DSETP.GT.AND P1, PT, |R12|, |R10|, PT;                     /* 0x1a0e000028c3dcc1 */
        /*0bb0*/                   SEL R15, R15, R14, P1;                                     /* 0x2002000038f3dc04 */
        /*0bb8*/                   IADD32I R12.CC, R15, 0x1;                                  /* 0x0c00000004f31c02 */
                                                                                              /* 0x22020042c20042b7 */
        /*0bc8*/                   LOP32I.AND R8, R15, 0x1;                                   /* 0x3800000004f21c02 */
        /*0bd0*/                   ISETP.EQ.U32.AND P1, PT, R8, 0x1, PT;                      /* 0x190ec0000483dc03 */
        /*0bd8*/                   IADD.X R13, RZ, R9;                                        /* 0x4800000027f35c43 */
        /*0be0*/                   IADD32I R10.CC, R15, -0x1;                                 /* 0x0ffffffffcf29c02 */
        /*0be8*/                   SEL R12, R15, R12, !P1;                                    /* 0x2012000030f31c04 */
        /*0bf0*/                   IADD32I.X R8, R9, -0x1;                                    /* 0x0bfffffffc921c42 */
        /*0bf8*/                   SEL R13, R9, R13, !P1;                                     /* 0x2012000034935c04 */
                                                                                              /* 0x22d2923282020267 */
        /*0c08*/                   SEL R10, R10, R15, !P1;                                    /* 0x201200003ca29c04 */
        /*0c10*/                   SEL R11, R8, R9, !P1;                                      /* 0x201200002482dc04 */
        /*0c18*/                   MOV R8, R12;                                               /* 0x2800000030021de4 */
        /*0c20*/                   MOV R9, R13;                                               /* 0x2800000034025de4 */
        /*0c28*/                   DFMA R8, R6, R8, -R4;                                      /* 0x2008000020621d01 */
        /*0c30*/                   DFMA R4, R6, R10, -R4;                                     /* 0x2008000028611d01 */
        /*0c38*/                   DSETP.GT.AND P1, PT, |R8|, |R4|, PT;                       /* 0x1a0e00001083dcc1 */
                                                                                              /* 0x228042b2d2e04207 */
        /*0c48*/                   SEL R8, R10, R12, P1;                                      /* 0x2002000030a21c04 */
        /*0c50*/                   SEL R9, R11, R13, P1;                                      /* 0x2002000034b25c04 */
        /*0c58*/                   RET;                                                       /* 0x9000000000001de7 */
        /*0c60*/                   DSETP.EQ.AND P1, PT, R14, RZ, PT;                          /* 0x190e0000fce3dc01 */
        /*0c68*/               @P1 BRA 0xcd0;                                                 /* 0x40000001800005e7 */
        /*0c70*/                   MUFU.RCP64H R9, R7;                                        /* 0xc800000018725c00 */
        /*0c78*/                   MOV R8, RZ;                                                /* 0x28000000fc021de4 */
                                                                                              /* 0x220282027202d2d7 */
        /*0c88*/                   DSETP.GT.AND P2, PT, |R8|, RZ, PT;                         /* 0x1a0e0000fc85dc81 */
        /*0c90*/              @!P2 DSETP.NEU.AND P1, PT, |R6|, +INF , PT;                     /* 0x1e8edffc0063e881 */
        /*0c98*/              @!P2 SEL R6, R6, R8, P1;                                        /* 0x200200002061a804 */
        /*0ca0*/              @!P2 SEL R7, R7, R9, P1;                                        /* 0x200200002471e804 */
        /*0ca8*/              @!P2 MOV R8, R6;                                                /* 0x28000000180229e4 */
        /*0cb0*/              @!P2 MOV R9, R7;                                                /* 0x280000001c0269e4 */
        /*0cb8*/                   DMUL R8, R4, R8;                                           /* 0x5000000020421c01 */
                                                                                              /* 0x2000000002e202e7 */
        /*0cc8*/                   RET;                                                       /* 0x9000000000001de7 */
        /*0cd0*/                   DMUL R8, R4, R6;                                           /* 0x5000000018421c01 */
        /*0cd8*/                   RET;                                                       /* 0x9000000000001de7 */
        /*0ce0*/                   BRA 0xce0;                                                 /* 0x4003ffffe0001de7 */
        /*0ce8*/                   NOP;                                                       /* 0x4000000000001de4 */
        /*0cf0*/                   NOP;                                                       /* 0x4000000000001de4 */
        /*0cf8*/                   NOP;                                                       /* 0x4000000000001de4 */
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
        /*0020*/                   MOV R5, c[0x0][0x140];                                      /* 0x2800400500015de4 */
        /*0028*/                   MOV32I R6, 0x1;                                             /* 0x1800000004019de2 */
        /*0030*/                   IADD32I RZ.CC, R0, -0x20;                                   /* 0x0fffffff800fdc02 */
        /*0038*/                   SHL R3, R5, -0x1d;                                          /* 0x6000ffff8c50dc03 */
                                                                                               /* 0x2282b04202423047 */
        /*0048*/                   MOV32I R0, 0x0;                                             /* 0x1800000000001de2 */
        /*0050*/                   SHL R2, R2, 0x3;                                            /* 0x6000c0000c209c03 */
        /*0058*/                   SHR.U32 R4, R5, 0x1d;                                       /* 0x5800c00074511c03 */
        /*0060*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*0068*/                   IMAD.U32.U32 R5.CC, R5, 0x8, R0;                            /* 0x2001c00020515c03 */
        /*0070*/                   MOV32I R0, 0x0;                                             /* 0x1800000000001de2 */
        /*0078*/               @P0 IADD R3, R2, R4;                                            /* 0x480000001020c003 */
                                                                                               /* 0x22329283f2328047 */
        /*0088*/                   MOV R2, R5;                                                 /* 0x2800000014009de4 */
        /*0090*/                   IADD.X R3, R0, R3;                                          /* 0x480000000c00dc43 */
        /*0098*/                   LD.E.64 R8, [R2+0x10];                                      /* 0x8400000040221ca5 */
        /*00a0*/                   LD.E.64 R4, [R2+0x4e210];                                   /* 0x8400138840211ca5 */
        /*00a8*/                   MUFU.RCP64H R7, R9;                                         /* 0xc80000001891dc00 */
        /*00b0*/                   DFMA R10, -R8, R6, c[0x2][0x0];                             /* 0x200c880000829e01 */
        /*00b8*/                   DFMA R10, R10, R10, R10;                                    /* 0x2014000028a29c01 */
                                                                                               /* 0x2282820292929257 */
        /*00c8*/                   DMUL R14, R4, c[0x0][0x148];                                /* 0x5000400520439c01 */
        /*00d0*/                   DFMA R10, R6, R10, R6;                                      /* 0x200c000028629c01 */
        /*00d8*/                   DMUL R12, R14, R10;                                         /* 0x5000000028e31c01 */
        /*00e0*/                   DFMA R6, -R8, R12, R14;                                     /* 0x201c000030819e01 */
        /*00e8*/                   DFMA R12, R10, R6, R12;                                     /* 0x2018000018a31c01 */
        /*00f0*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                 /* 0x160f480020ffdc80 */
        /*00f8*/                   FFMA R0, RZ, R9, R13;                                       /* 0x301a000027f01c00 */
                                                                                               /* 0x22b042b2b042b2c7 */
        /*0108*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0xc], PT;                 /* 0x220e48003001dc80 */
        /*0110*/               @P0 BRA CC.NEU, 0x140;                                          /* 0x40000000a00001a7 */
        /*0118*/                   MOV R10, R8;                                                /* 0x2800000020029de4 */
        /*0120*/                   MOV R11, R9;                                                /* 0x280000002402dde4 */
        /*0128*/                   CAL 0x2e8;                                                  /* 0x50000006e0011c07 */
        /*0130*/                   MOV R12, R22;                                               /* 0x2800000058031de4 */
        /*0138*/                   MOV R13, R23;                                               /* 0x280000005c035de4 */
                                                                                               /* 0x23f2323232323047 */
        /*0148*/                   LD.E.64 R10, [R2+0x10];                                     /* 0x8400000040229ca5 */
        /*0150*/                   MOV32I R20, 0x1;                                            /* 0x1800000004051de2 */
        /*0158*/                   LD.E.64 R16, [R2+0x5dc10];                                  /* 0x8400177040241ca5 */
        /*0160*/                   LD.E.64 R14, [R2+0x2ee10];                                  /* 0x84000bb840239ca5 */
        /*0168*/                   LD.E.64 R4, [R2+0x3e810];                                   /* 0x84000fa040211ca5 */
        /*0170*/                   LD.E.64 R6, [R2+0xfa10];                                    /* 0x840003e840219ca5 */
        /*0178*/                   LD.E.64 R8, [R2+0x1f410];                                   /* 0x840007d040221ca5 */
                                                                                               /* 0x2292929252329287 */
        /*0188*/                   MUFU.RCP64H R21, R11;                                       /* 0xc800000018b55c00 */
        /*0190*/                   DFMA R18, -R10, R20, c[0x2][0x0];                           /* 0x2028880000a49e01 */
        /*0198*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*01a0*/                   DMUL R16, R16, c[0x0][0x148];                               /* 0x5000400521041c01 */
        /*01a8*/                   DFMA R18, R20, R18, R20;                                    /* 0x2028000049449c01 */
        /*01b0*/                   DMUL R20, R16, R18;                                         /* 0x5000000049051c01 */
        /*01b8*/                   DFMA R22, -R10, R20, R16;                                   /* 0x2020000050a59e01 */
                                                                                               /* 0x22b2c27200429237 */
        /*01c8*/                   DFMA R22, R18, R22, R20;                                    /* 0x2028000059259c01 */
        /*01d0*/                   DADD R12, R12, R14;                                         /* 0x4800000038c31c01 */
        /*01d8*/                   FFMA R0, RZ, R11, R23;                                      /* 0x302e00002ff01c00 */
        /*01e0*/                   ST.E.64 [R2+0x2ee10], R12;                                  /* 0x94000bb840231ca5 */
        /*01e8*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                 /* 0x160f4800211fdc80 */
        /*01f0*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0xc], PT;                 /* 0x220e48003001dc80 */
        /*01f8*/               @P0 BRA CC.NEU, 0x220;                                          /* 0x40000000800001a7 */
                                                                                               /* 0x22e2e25232b2b047 */
        /*0208*/                   MOV R14, R16;                                               /* 0x2800000040039de4 */
        /*0210*/                   MOV R15, R17;                                               /* 0x280000004403dde4 */
        /*0218*/                   CAL 0x2e8;                                                  /* 0x5000000320011c07 */
        /*0220*/                   DFMA R6, R12, c[0x0][0x148], R6;                            /* 0x200c400520c19c01 */
        /*0228*/                   DADD R4, R4, R22;                                           /* 0x4800000058411c01 */
        /*0230*/                   ST.E.64 [R2+0xfa10], R6;                                    /* 0x940003e840219ca5 */
        /*0238*/                   LD.E.64 R12, [R2+0xfa10];                                   /* 0x840003e840231ca5 */
                                                                                               /* 0x22b04292f2d3f2e7 */
        /*0248*/                   ST.E.64 [R2+0x3e810], R4;                                   /* 0x94000fa040211ca5 */
        /*0250*/                   LD.E.64 R10, [R2+0x3e810];                                  /* 0x84000fa040229ca5 */
        /*0258*/                   DSETP.GT.AND P0, PT, R12, 1, PT;                            /* 0x1a0ecffc00c1dc01 */
        /*0260*/                   DSETP.LT.OR P0, PT, R12, -1, P0;                            /* 0x18a0effc00c1dc01 */
        /*0268*/                   DFMA R8, R10, c[0x0][0x148], R8;                            /* 0x2010400520a21c01 */
        /*0270*/                   ST.E.64 [R2+0x1f410], R8;                                   /* 0x940007d040221ca5 */
        /*0278*/              @!P0 BRA 0x2a8;                                                  /* 0x40000000a00021e7 */
                                                                                               /* 0x22e2d2d2b283f237 */
        /*0288*/                   LD.E.64 R4, [R2+0x2ee10];                                   /* 0x84000bb840211ca5 */
        /*0290*/                   LD.E.64 R8, [R2+0x1f410];                                   /* 0x840007d040221ca5 */
        /*0298*/                   F2F.F64.F64 R4, -R4;                                        /* 0x1000000011b11d04 */
        /*02a0*/                   ST.E.64 [R2+0x2ee10], R4;                                   /* 0x94000bb840211ca5 */
        /*02a8*/                   DSETP.GT.AND P0, PT, R8, 1, PT;                             /* 0x1a0ecffc0081dc01 */
        /*02b0*/                   DSETP.LT.OR P0, PT, R8, -1, P0;                             /* 0x18a0effc0081dc01 */
        /*02b8*/              @!P0 EXIT;                                                       /* 0x80000000000021e7 */
                                                                                               /* 0x22028042e04283f7 */
        /*02c8*/                   LD.E.64 R4, [R2+0x3e810];                                   /* 0x84000fa040211ca5 */
        /*02d0*/                   F2F.F64.F64 R4, -R4;                                        /* 0x1000000011b11d04 */
        /*02d8*/                   ST.E.64 [R2+0x3e810], R4;                                   /* 0x94000fa040211ca5 */
        /*02e0*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*02e8*/                   LOP32I.AND R0, R11, 0x40000000;                             /* 0x3900000000b01c02 */
        /*02f0*/                   LOP32I.AND R26, R15, 0x7f800000;                            /* 0x39fe000000f69c02 */
        /*02f8*/                   ISETP.LT.U32.AND P0, PT, R0, c[0x2][0x10], PT;              /* 0x188e48004001dc03 */
                                                                                               /* 0x22c042028042b047 */
        /*0308*/                   MOV32I R0, 0x1ff00000;                                      /* 0x187fc00000001de2 */
        /*0310*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*0318*/                   SEL R17, R0, c[0x2][0x14], !P0;                             /* 0x2010480050045c04 */
        /*0320*/                   MOV32I R20, 0x1;                                            /* 0x1800000004051de2 */
        /*0328*/                   DMUL R18, R10, R16;                                         /* 0x5000000040a49c01 */
        /*0330*/                   ISETP.LT.U32.AND P0, PT, R26, c[0x2][0x18], PT;             /* 0x188e480061a1dc03 */
        /*0338*/                   MOV R24, RZ;                                                /* 0x28000000fc061de4 */
                                                                                               /* 0x2292925232928047 */
        /*0348*/                   MUFU.RCP64H R21, R19;                                       /* 0xc800000019355c00 */
        /*0350*/                   SEL R25, R0, c[0x2][0x14], !P0;                             /* 0x2010480050065c04 */
        /*0358*/                   DFMA R22, -R18, R20, c[0x2][0x0];                           /* 0x2028880001259e01 */
        /*0360*/                   DFMA R30, R22, R22, R22;                                    /* 0x202c000059679c01 */
        /*0368*/                   DMUL R22, R14, R24;                                         /* 0x5000000060e59c01 */
        /*0370*/                   DFMA R20, R20, R30, R20;                                    /* 0x2028000079451c01 */
        /*0378*/                   DMUL R24, R22, R20;                                         /* 0x5000000051661c01 */
                                                                                               /* 0x22823042b2d29297 */
        /*0388*/                   DFMA R18, -R18, R24, R22;                                   /* 0x202c000061249e01 */
        /*0390*/                   DFMA R22, R20, R18, R24;                                    /* 0x2030000049459c01 */
        /*0398*/                   DSETP.LEU.AND P0, PT, |R22|, RZ, PT;                        /* 0x1d8e0000fd61dc81 */
        /*03a0*/               @P0 BRA 0x558;                                                  /* 0x40000006c00001e7 */
        /*03a8*/                   ISETP.GT.U32.AND P0, PT, R26, c[0x2][0x1c], PT;             /* 0x1a0e480071a1dc03 */
        /*03b0*/                   MOV R18, RZ;                                                /* 0x28000000fc049de4 */
        /*03b8*/                   DMUL R20, R16, R22;                                         /* 0x5000000059051c01 */
                                                                                               /* 0x22d2925232523287 */
        /*03c8*/                   SEL R19, R0, c[0x2][0x14], !P0;                             /* 0x201048005004dc04 */
        /*03d0*/                   DMUL R22, R22, R18;                                         /* 0x5000000049659c01 */
        /*03d8*/                   DMUL R20, R18, R20;                                         /* 0x5000000051251c01 */
        /*03e0*/                   DMUL R16, R16, R22;                                         /* 0x5000000059041c01 */
        /*03e8*/                   DFMA R18, R10, R20, -R14;                                   /* 0x201c000050a49d01 */
        /*03f0*/                   DFMA R22, R10, R16, -R14;                                   /* 0x201c000040a59d01 */
        /*03f8*/                   DSETP.GT.AND P0, PT, |R18|, |R22|, PT;                      /* 0x1a0e00005921dcc1 */
                                                                                               /* 0x22220042b2c27207 */
        /*0408*/                   SEL R17, R17, R21, P0;                                      /* 0x2000000055145c04 */
        /*0410*/                   SEL R16, R16, R20, P0;                                      /* 0x2000000051041c04 */
        /*0418*/                   FSETP.GTU.AND P0, PT, |R17|, 1.469367938527859385e-39, PT;  /* 0x260ec0040111dc80 */
        /*0420*/               @P0 BRA 0x5d0;                                                  /* 0x40000006a00001e7 */
        /*0428*/                   FSETP.LT.AND P0, PT, |R15|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000f1dc80 */
        /*0430*/                   LOP32I.AND R0, R16, 0xfffffffe;                             /* 0x3bfffffff9001c02 */
        /*0438*/                   MOV32I R21, 0x3ff00000;                                     /* 0x18ffc00000055de2 */
                                                                                               /* 0x2202023280428047 */
        /*0448*/                   MOV R20, RZ;                                                /* 0x28000000fc051de4 */
        /*0450*/                   LOP32I.OR R22, R16, 0x1;                                    /* 0x3800000005059c42 */
        /*0458*/                   SEL R21, R21, c[0x2][0x20], !P0;                            /* 0x2010480081555c04 */
        /*0460*/                   MOV R16, R0;                                                /* 0x2800000000041de4 */
        /*0468*/                   DMUL R10, R10, R20;                                         /* 0x5000000050a29c01 */
        /*0470*/                   DMUL R14, R14, R20;                                         /* 0x5000000050e39c01 */
        /*0478*/                   MOV R20, R22;                                               /* 0x2800000058051de4 */
                                                                                               /* 0x22c04282d2923277 */
        /*0488*/                   MOV R21, R17;                                               /* 0x2800000044055de4 */
        /*0490*/                   DFMA R18, R16, R10, -R14;                                   /* 0x201c000029049d01 */
        /*0498*/                   DFMA R20, R20, R10, -R14;                                   /* 0x201c000029451d01 */
        /*04a0*/                   DSETP.GT.AND P0, PT, |R18|, |R20|, PT;                      /* 0x1a0e00005121dcc1 */
        /*04a8*/                   SEL R23, R22, R0, P0;                                       /* 0x200000000165dc04 */
        /*04b0*/                   IADD32I R0.CC, R23, 0x1;                                    /* 0x0c00000005701c02 */
        /*04b8*/                   LOP32I.AND R18, R23, 0x1;                                   /* 0x3800000005749c02 */
                                                                                               /* 0x2262020042c20047 */
        /*04c8*/                   ISETP.EQ.U32.AND P0, PT, R18, 0x1, PT;                      /* 0x190ec0000521dc03 */
        /*04d0*/                   IADD.X R16, RZ, R17;                                        /* 0x4800000047f41c43 */
        /*04d8*/                   IADD32I R20.CC, R23, -0x1;                                  /* 0x0ffffffffd751c02 */
        /*04e0*/                   SEL R0, R23, R0, !P0;                                       /* 0x2010000001701c04 */
        /*04e8*/                   IADD32I.X R21, R17, -0x1;                                   /* 0x0bfffffffd155c42 */
        /*04f0*/                   SEL R19, R17, R16, !P0;                                     /* 0x201000004114dc04 */
        /*04f8*/                   SEL R20, R20, R23, !P0;                                     /* 0x201000005d451c04 */
                                                                                               /* 0x2202d29232820207 */
        /*0508*/                   SEL R21, R21, R17, !P0;                                     /* 0x2010000045555c04 */
        /*0510*/                   MOV R16, R0;                                                /* 0x2800000000041de4 */
        /*0518*/                   MOV R17, R19;                                               /* 0x280000004c045de4 */
        /*0520*/                   DFMA R16, R10, R16, -R14;                                   /* 0x201c000040a41d01 */
        /*0528*/                   DFMA R10, R10, R20, -R14;                                   /* 0x201c000050a29d01 */
        /*0530*/                   DSETP.GT.AND P0, PT, |R16|, |R10|, PT;                      /* 0x1a0e00002901dcc1 */
        /*0538*/                   SEL R16, R20, R0, P0;                                       /* 0x2000000001441c04 */
                                                                                               /* 0x22d28042b2d2e047 */
        /*0548*/                   SEL R17, R21, R19, P0;                                      /* 0x200000004d545c04 */
        /*0550*/                   BRA 0x5d0;                                                  /* 0x40000001e0001de7 */
        /*0558*/                   DSETP.EQ.AND P0, PT, R22, RZ, PT;                           /* 0x190e0000fd61dc01 */
        /*0560*/               @P0 BRA 0x5c0;                                                  /* 0x40000001600001e7 */
        /*0568*/                   MUFU.RCP64H R17, R11;                                       /* 0xc800000018b45c00 */
        /*0570*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*0578*/                   DSETP.GT.AND P1, PT, |R16|, RZ, PT;                         /* 0x1a0e0000fd03dc81 */
                                                                                               /* 0x22e20282027202d7 */
        /*0588*/              @!P1 DSETP.NEU.AND P0, PT, |R10|, +INF , PT;                     /* 0x1e8edffc00a1e481 */
        /*0590*/              @!P1 SEL R0, R10, R16, P0;                                       /* 0x2000000040a02404 */
        /*0598*/              @!P1 SEL R11, R11, R17, P0;                                      /* 0x2000000044b2e404 */
        /*05a0*/              @!P1 MOV R16, R0;                                                /* 0x28000000000425e4 */
        /*05a8*/              @!P1 MOV R17, R11;                                               /* 0x280000002c0465e4 */
        /*05b0*/                   DMUL R16, R14, R16;                                         /* 0x5000000040e41c01 */
        /*05b8*/                   BRA 0x5d0;                                                  /* 0x4000000040001de7 */
                                                                                               /* 0x20000002e20042b7 */
        /*05c8*/                   DMUL R16, R14, R10;                                         /* 0x5000000028e41c01 */
        /*05d0*/                   MOV R22, R16;                                               /* 0x2800000040059de4 */
        /*05d8*/                   MOV R23, R17;                                               /* 0x280000004405dde4 */
        /*05e0*/                   RET;                                                        /* 0x9000000000001de7 */
        /*05e8*/                   BRA 0x5e8;                                                  /* 0x4003ffffe0001de7 */
        /*05f0*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*05f8*/                   NOP;                                                        /* 0x4000000000001de4 */
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
        /*04b8*/                   MOV32I R24, 0x1ff00000;                                     /* 0x187fc00000061de2 */
                                                                                               /* 0x22c042028042c047 */
        /*04c8*/                   ISETP.LT.U32.AND P0, PT, R14, c[0x2][0x24], PT;             /* 0x188e480090e1dc03 */
        /*04d0*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
        /*04d8*/                   SEL R15, R24, c[0x2][0x28], !P0;                            /* 0x20104800a183dc04 */
        /*04e0*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
        /*04e8*/                   DMUL R16, R12, R14;                                         /* 0x5000000038c41c01 */
        /*04f0*/                   ISETP.LT.U32.AND P0, PT, R26, c[0x2][0x2c], PT;             /* 0x188e4800b1a1dc03 */
        /*04f8*/                   MOV R28, RZ;                                                /* 0x28000000fc071de4 */
                                                                                               /* 0x2292925232928047 */
        /*0508*/                   MUFU.RCP64H R19, R17;                                       /* 0xc80000001914dc00 */
        /*0510*/                   SEL R29, R24, c[0x2][0x28], !P0;                            /* 0x20104800a1875c04 */
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
        /*0588*/                   SEL R17, R24, c[0x2][0x28], !P0;                            /* 0x20104800a1845c04 */
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
        /*05f8*/                   MOV32I R17, 0x3ff00000;                                     /* 0x18ffc00000045de2 */
                                                                                               /* 0x2202023280428047 */
        /*0608*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*0610*/                   LOP32I.OR R27, R14, 0x1;                                    /* 0x3800000004e6dc42 */
        /*0618*/                   SEL R17, R17, c[0x2][0x34], !P0;                            /* 0x20104800d1145c04 */
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
        /*07a8*/                   ISETP.LT.U32.AND P0, PT, R12, c[0x2][0x38], PT;             /* 0x188e4800e0c1dc03 */
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
        /*0850*/                   ISETP.GT.AND P0, PT, R7, c[0x2][0x3c], PT;                  /* 0x1a0e4800f071dc23 */
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
.reg .b64 %rd<18>;


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
add.s64 %rd11, %rd6, 128016;
add.s64 %rd12, %rd11, %rd5;
add.s64 %rd13, %rd11, %rd9;
ld.global.f64 %fd4, [%rd13];
ld.global.f64 %fd5, [%rd12];
sub.f64 %fd6, %fd5, %fd4;
mul.f64 %fd7, %fd6, %fd6;
fma.rn.f64 %fd8, %fd3, %fd3, %fd7;
sqrt.rn.f64 %fd9, %fd8;
add.s64 %rd14, %rd6, 16;
add.s64 %rd15, %rd14, %rd9;
ld.global.f64 %fd10, [%rd15];
mul.f64 %fd11, %fd10, 0d3DD257B4D002790D;
add.s64 %rd16, %rd14, %rd5;
ld.global.f64 %fd12, [%rd16];
mul.f64 %fd13, %fd11, %fd12;
fma.rn.f64 %fd14, %fd9, %fd9, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd15, %fd13, %fd14;
mul.f64 %fd16, %fd3, %fd15;
div.rn.f64 %fd17, %fd16, %fd9;
add.s64 %rd17, %rd6, %rd9;
ld.global.f64 %fd18, [%rd17+320016];
add.f64 %fd19, %fd17, %fd18;
ld.global.f64 %fd20, [%rd17+384016];
st.global.f64 [%rd17+320016], %fd19;
mul.f64 %fd21, %fd6, %fd15;
div.rn.f64 %fd22, %fd21, %fd9;
add.f64 %fd23, %fd22, %fd20;
st.global.f64 [%rd17+384016], %fd23;

BB1_2:
ret;
}


.visible .entry _Z14codegen_updateP4Bodyd(
.param .u64 _Z14codegen_updateP4Bodyd_param_0,
.param .f64 _Z14codegen_updateP4Bodyd_param_1
)
{
.reg .pred %p<7>;
.reg .f64 %fd<27>;
.reg .b64 %rd<8>;


ld.param.u64 %rd4, [_Z14codegen_updateP4Bodyd_param_0];
ld.param.f64 %fd4, [_Z14codegen_updateP4Bodyd_param_1];
shl.b64 %rd5, %rd4, 3;
mov.u64 %rd6, __Bodydata_buffer;
add.s64 %rd7, %rd6, %rd5;
ld.global.f64 %fd5, [%rd7+320016];
mul.f64 %fd6, %fd5, %fd4;
ld.global.f64 %fd7, [%rd7+16];
div.rn.f64 %fd8, %fd6, %fd7;
add.s64 %rd1, %rd7, 192016;
ld.global.f64 %fd9, [%rd7+192016];
ld.global.f64 %fd10, [%rd7+384016];
ld.global.f64 %fd11, [%rd7+256016];
add.f64 %fd12, %fd8, %fd9;
ld.global.f64 %fd13, [%rd7+64016];
ld.global.f64 %fd14, [%rd7+128016];
st.global.f64 [%rd7+192016], %fd12;
mul.f64 %fd15, %fd10, %fd4;
ld.global.f64 %fd16, [%rd7+16];
div.rn.f64 %fd17, %fd15, %fd16;
add.s64 %rd2, %rd7, 256016;
add.f64 %fd18, %fd17, %fd11;
st.global.f64 [%rd7+256016], %fd18;
fma.rn.f64 %fd19, %fd12, %fd4, %fd13;
st.global.f64 [%rd7+64016], %fd19;
ld.global.f64 %fd20, [%rd7+256016];
add.s64 %rd3, %rd7, 128016;
fma.rn.f64 %fd26, %fd20, %fd4, %fd14;
ld.global.f64 %fd21, [%rd7+64016];
st.global.f64 [%rd7+128016], %fd26;
setp.lt.f64	%p1, %fd21, 0dBFF0000000000000;
setp.gt.f64	%p2, %fd21, 0d3FF0000000000000;
or.pred %p3, %p1, %p2;
@!%p3 bra BB2_2;
bra.uni BB2_1;

BB2_1:
ld.global.f64 %fd22, [%rd1];
neg.f64 %fd23, %fd22;
st.global.f64 [%rd1], %fd23;
ld.global.f64 %fd26, [%rd3];

BB2_2:
setp.gt.f64	%p4, %fd26, 0d3FF0000000000000;
setp.lt.f64	%p5, %fd26, 0dBFF0000000000000;
or.pred %p6, %p5, %p4;
@!%p6 bra BB2_4;
bra.uni BB2_3;

BB2_3:
ld.global.f64 %fd24, [%rd2];
neg.f64 %fd25, %fd24;
st.global.f64 [%rd2], %fd25;

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
.reg .f64 %fd<92>;
.reg .b64 %rd<12>;


ld.param.u64 %rd8, [_Z22codengen_simple_updateP4Bodyd_param_0];
ld.param.f64 %fd1, [_Z22codengen_simple_updateP4Bodyd_param_1];
shl.b64 %rd9, %rd8, 3;
mov.u64 %rd10, __Bodydata_buffer;
add.s64 %rd11, %rd10, %rd9;
add.s64 %rd1, %rd11, 320016;
add.s64 %rd2, %rd11, 16;
add.s64 %rd3, %rd11, 192016;
add.s64 %rd4, %rd11, 384016;
add.s64 %rd5, %rd11, 256016;
add.s64 %rd6, %rd11, 64016;
add.s64 %rd7, %rd11, 128016;
mov.u32 %r4, -1000;

BB6_1:
ld.global.f64 %fd2, [%rd1];
mul.f64 %fd3, %fd2, %fd1;
ld.global.f64 %fd4, [%rd2];
div.rn.f64 %fd5, %fd3, %fd4;
ld.global.f64 %fd6, [%rd3];
add.f64 %fd7, %fd5, %fd6;
st.global.f64 [%rd3], %fd7;
ld.global.f64 %fd8, [%rd4];
mul.f64 %fd9, %fd8, %fd1;
ld.global.f64 %fd10, [%rd2];
div.rn.f64 %fd11, %fd9, %fd10;
ld.global.f64 %fd12, [%rd5];
add.f64 %fd13, %fd11, %fd12;
st.global.f64 [%rd5], %fd13;
ld.global.f64 %fd14, [%rd3];
ld.global.f64 %fd15, [%rd6];
fma.rn.f64 %fd16, %fd14, %fd1, %fd15;
st.global.f64 [%rd6], %fd16;
ld.global.f64 %fd17, [%rd5];
ld.global.f64 %fd18, [%rd7];
fma.rn.f64 %fd19, %fd17, %fd1, %fd18;
st.global.f64 [%rd7], %fd19;
ld.global.f64 %fd20, [%rd1];
mul.f64 %fd21, %fd20, %fd1;
ld.global.f64 %fd22, [%rd2];
div.rn.f64 %fd23, %fd21, %fd22;
ld.global.f64 %fd24, [%rd3];
add.f64 %fd25, %fd23, %fd24;
st.global.f64 [%rd3], %fd25;
ld.global.f64 %fd26, [%rd4];
mul.f64 %fd27, %fd26, %fd1;
ld.global.f64 %fd28, [%rd2];
div.rn.f64 %fd29, %fd27, %fd28;
ld.global.f64 %fd30, [%rd5];
add.f64 %fd31, %fd29, %fd30;
st.global.f64 [%rd5], %fd31;
ld.global.f64 %fd32, [%rd3];
ld.global.f64 %fd33, [%rd6];
fma.rn.f64 %fd34, %fd32, %fd1, %fd33;
st.global.f64 [%rd6], %fd34;
ld.global.f64 %fd35, [%rd5];
ld.global.f64 %fd36, [%rd7];
fma.rn.f64 %fd37, %fd35, %fd1, %fd36;
st.global.f64 [%rd7], %fd37;
ld.global.f64 %fd38, [%rd1];
mul.f64 %fd39, %fd38, %fd1;
ld.global.f64 %fd40, [%rd2];
div.rn.f64 %fd41, %fd39, %fd40;
ld.global.f64 %fd42, [%rd3];
add.f64 %fd43, %fd41, %fd42;
st.global.f64 [%rd3], %fd43;
ld.global.f64 %fd44, [%rd4];
mul.f64 %fd45, %fd44, %fd1;
ld.global.f64 %fd46, [%rd2];
div.rn.f64 %fd47, %fd45, %fd46;
ld.global.f64 %fd48, [%rd5];
add.f64 %fd49, %fd47, %fd48;
st.global.f64 [%rd5], %fd49;
ld.global.f64 %fd50, [%rd3];
ld.global.f64 %fd51, [%rd6];
fma.rn.f64 %fd52, %fd50, %fd1, %fd51;
st.global.f64 [%rd6], %fd52;
ld.global.f64 %fd53, [%rd5];
ld.global.f64 %fd54, [%rd7];
fma.rn.f64 %fd55, %fd53, %fd1, %fd54;
st.global.f64 [%rd7], %fd55;
ld.global.f64 %fd56, [%rd1];
mul.f64 %fd57, %fd56, %fd1;
ld.global.f64 %fd58, [%rd2];
div.rn.f64 %fd59, %fd57, %fd58;
ld.global.f64 %fd60, [%rd3];
add.f64 %fd61, %fd59, %fd60;
st.global.f64 [%rd3], %fd61;
ld.global.f64 %fd62, [%rd4];
mul.f64 %fd63, %fd62, %fd1;
ld.global.f64 %fd64, [%rd2];
div.rn.f64 %fd65, %fd63, %fd64;
ld.global.f64 %fd66, [%rd5];
add.f64 %fd67, %fd65, %fd66;
st.global.f64 [%rd5], %fd67;
ld.global.f64 %fd68, [%rd3];
ld.global.f64 %fd69, [%rd6];
fma.rn.f64 %fd70, %fd68, %fd1, %fd69;
st.global.f64 [%rd6], %fd70;
ld.global.f64 %fd71, [%rd5];
ld.global.f64 %fd72, [%rd7];
fma.rn.f64 %fd73, %fd71, %fd1, %fd72;
st.global.f64 [%rd7], %fd73;
ld.global.f64 %fd74, [%rd1];
mul.f64 %fd75, %fd74, %fd1;
ld.global.f64 %fd76, [%rd2];
div.rn.f64 %fd77, %fd75, %fd76;
ld.global.f64 %fd78, [%rd3];
add.f64 %fd79, %fd77, %fd78;
st.global.f64 [%rd3], %fd79;
ld.global.f64 %fd80, [%rd4];
mul.f64 %fd81, %fd80, %fd1;
ld.global.f64 %fd82, [%rd2];
div.rn.f64 %fd83, %fd81, %fd82;
ld.global.f64 %fd84, [%rd5];
add.f64 %fd85, %fd83, %fd84;
st.global.f64 [%rd5], %fd85;
ld.global.f64 %fd86, [%rd3];
ld.global.f64 %fd87, [%rd6];
fma.rn.f64 %fd88, %fd86, %fd1, %fd87;
st.global.f64 [%rd6], %fd88;
ld.global.f64 %fd89, [%rd5];
ld.global.f64 %fd90, [%rd7];
fma.rn.f64 %fd91, %fd89, %fd1, %fd90;
st.global.f64 [%rd7], %fd91;
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
.reg .b64 %rd<91>;


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
mov.u64 %rd89, __Bodydata_buffer;
add.s64 %rd28, %rd27, %rd89;
add.s64 %rd3, %rd28, 320024;
add.s64 %rd4, %rd28, 384024;
mov.u64 %rd90, 8000;
mov.u64 %rd88, 5;
mov.u64 %rd87, 4;
mov.u64 %rd86, 3;
mov.u64 %rd85, 2;
mov.u64 %rd84, 1;

BB8_2:
setp.eq.s64	%p2, %rd84, %rd2;
@%p2 bra BB8_4;

shl.b64 %rd29, %rd84, 3;
mov.u64 %rd30, __Bodydata_buffer;
add.s64 %rd31, %rd30, 64016;
add.s64 %rd32, %rd31, %rd29;
mul.wide.s32 %rd33, %r4, 8;
add.s64 %rd34, %rd33, 8;
add.s64 %rd35, %rd31, %rd34;
ld.global.f64 %fd1, [%rd35];
ld.global.f64 %fd2, [%rd32];
sub.f64 %fd3, %fd2, %fd1;
add.s64 %rd36, %rd30, 128016;
add.s64 %rd37, %rd36, %rd29;
add.s64 %rd38, %rd36, %rd34;
ld.global.f64 %fd4, [%rd38];
ld.global.f64 %fd5, [%rd37];
sub.f64 %fd6, %fd5, %fd4;
mul.f64 %fd7, %fd6, %fd6;
fma.rn.f64 %fd8, %fd3, %fd3, %fd7;
sqrt.rn.f64 %fd9, %fd8;
add.s64 %rd39, %rd30, %rd33;
ld.global.f64 %fd10, [%rd39+24];
mul.f64 %fd11, %fd10, 0d3DD257B4D002790D;
ld.global.f64 %fd12, [%rd89+24];
mul.f64 %fd13, %fd11, %fd12;
fma.rn.f64 %fd14, %fd9, %fd9, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd15, %fd13, %fd14;
mul.f64 %fd16, %fd3, %fd15;
div.rn.f64 %fd17, %fd16, %fd9;
ld.global.f64 %fd18, [%rd3];
add.f64 %fd19, %fd17, %fd18;
st.global.f64 [%rd3], %fd19;
mul.f64 %fd20, %fd6, %fd15;
div.rn.f64 %fd21, %fd20, %fd9;
ld.global.f64 %fd22, [%rd4];
add.f64 %fd23, %fd21, %fd22;
st.global.f64 [%rd4], %fd23;

BB8_4:
setp.eq.s64	%p3, %rd85, %rd2;
@%p3 bra BB8_6;

shl.b64 %rd40, %rd85, 3;
mov.u64 %rd41, __Bodydata_buffer;
add.s64 %rd42, %rd41, 64016;
add.s64 %rd43, %rd42, %rd40;
mul.wide.s32 %rd44, %r4, 8;
add.s64 %rd45, %rd44, 8;
add.s64 %rd46, %rd42, %rd45;
ld.global.f64 %fd24, [%rd46];
ld.global.f64 %fd25, [%rd43];
sub.f64 %fd26, %fd25, %fd24;
add.s64 %rd47, %rd41, 128016;
add.s64 %rd48, %rd47, %rd40;
add.s64 %rd49, %rd47, %rd45;
ld.global.f64 %fd27, [%rd49];
ld.global.f64 %fd28, [%rd48];
sub.f64 %fd29, %fd28, %fd27;
mul.f64 %fd30, %fd29, %fd29;
fma.rn.f64 %fd31, %fd26, %fd26, %fd30;
sqrt.rn.f64 %fd32, %fd31;
add.s64 %rd50, %rd41, %rd44;
ld.global.f64 %fd33, [%rd50+24];
mul.f64 %fd34, %fd33, 0d3DD257B4D002790D;
ld.global.f64 %fd35, [%rd89+32];
mul.f64 %fd36, %fd34, %fd35;
fma.rn.f64 %fd37, %fd32, %fd32, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd38, %fd36, %fd37;
mul.f64 %fd39, %fd26, %fd38;
div.rn.f64 %fd40, %fd39, %fd32;
ld.global.f64 %fd41, [%rd3];
add.f64 %fd42, %fd40, %fd41;
st.global.f64 [%rd3], %fd42;
mul.f64 %fd43, %fd29, %fd38;
div.rn.f64 %fd44, %fd43, %fd32;
ld.global.f64 %fd45, [%rd4];
add.f64 %fd46, %fd44, %fd45;
st.global.f64 [%rd4], %fd46;

BB8_6:
setp.eq.s64	%p4, %rd86, %rd2;
@%p4 bra BB8_8;

shl.b64 %rd51, %rd86, 3;
mov.u64 %rd52, __Bodydata_buffer;
add.s64 %rd53, %rd52, 64016;
add.s64 %rd54, %rd53, %rd51;
mul.wide.s32 %rd55, %r4, 8;
add.s64 %rd56, %rd55, 8;
add.s64 %rd57, %rd53, %rd56;
ld.global.f64 %fd47, [%rd57];
ld.global.f64 %fd48, [%rd54];
sub.f64 %fd49, %fd48, %fd47;
add.s64 %rd58, %rd52, 128016;
add.s64 %rd59, %rd58, %rd51;
add.s64 %rd60, %rd58, %rd56;
ld.global.f64 %fd50, [%rd60];
ld.global.f64 %fd51, [%rd59];
sub.f64 %fd52, %fd51, %fd50;
mul.f64 %fd53, %fd52, %fd52;
fma.rn.f64 %fd54, %fd49, %fd49, %fd53;
sqrt.rn.f64 %fd55, %fd54;
add.s64 %rd61, %rd52, %rd55;
ld.global.f64 %fd56, [%rd61+24];
mul.f64 %fd57, %fd56, 0d3DD257B4D002790D;
ld.global.f64 %fd58, [%rd89+40];
mul.f64 %fd59, %fd57, %fd58;
fma.rn.f64 %fd60, %fd55, %fd55, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd61, %fd59, %fd60;
mul.f64 %fd62, %fd49, %fd61;
div.rn.f64 %fd63, %fd62, %fd55;
ld.global.f64 %fd64, [%rd3];
add.f64 %fd65, %fd63, %fd64;
st.global.f64 [%rd3], %fd65;
mul.f64 %fd66, %fd52, %fd61;
div.rn.f64 %fd67, %fd66, %fd55;
ld.global.f64 %fd68, [%rd4];
add.f64 %fd69, %fd67, %fd68;
st.global.f64 [%rd4], %fd69;

BB8_8:
setp.eq.s64	%p5, %rd87, %rd2;
@%p5 bra BB8_10;

shl.b64 %rd62, %rd87, 3;
mov.u64 %rd63, __Bodydata_buffer;
add.s64 %rd64, %rd63, 64016;
add.s64 %rd65, %rd64, %rd62;
mul.wide.s32 %rd66, %r4, 8;
add.s64 %rd67, %rd66, 8;
add.s64 %rd68, %rd64, %rd67;
ld.global.f64 %fd70, [%rd68];
ld.global.f64 %fd71, [%rd65];
sub.f64 %fd72, %fd71, %fd70;
add.s64 %rd69, %rd63, 128016;
add.s64 %rd70, %rd69, %rd62;
add.s64 %rd71, %rd69, %rd67;
ld.global.f64 %fd73, [%rd71];
ld.global.f64 %fd74, [%rd70];
sub.f64 %fd75, %fd74, %fd73;
mul.f64 %fd76, %fd75, %fd75;
fma.rn.f64 %fd77, %fd72, %fd72, %fd76;
sqrt.rn.f64 %fd78, %fd77;
add.s64 %rd72, %rd63, %rd66;
ld.global.f64 %fd79, [%rd72+24];
mul.f64 %fd80, %fd79, 0d3DD257B4D002790D;
ld.global.f64 %fd81, [%rd89+48];
mul.f64 %fd82, %fd80, %fd81;
fma.rn.f64 %fd83, %fd78, %fd78, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd84, %fd82, %fd83;
mul.f64 %fd85, %fd72, %fd84;
div.rn.f64 %fd86, %fd85, %fd78;
ld.global.f64 %fd87, [%rd3];
add.f64 %fd88, %fd86, %fd87;
st.global.f64 [%rd3], %fd88;
mul.f64 %fd89, %fd75, %fd84;
div.rn.f64 %fd90, %fd89, %fd78;
ld.global.f64 %fd91, [%rd4];
add.f64 %fd92, %fd90, %fd91;
st.global.f64 [%rd4], %fd92;

BB8_10:
setp.eq.s64	%p6, %rd88, %rd2;
@%p6 bra BB8_12;

shl.b64 %rd73, %rd88, 3;
mov.u64 %rd74, __Bodydata_buffer;
add.s64 %rd75, %rd74, 64016;
add.s64 %rd76, %rd75, %rd73;
mul.wide.s32 %rd77, %r4, 8;
add.s64 %rd78, %rd77, 8;
add.s64 %rd79, %rd75, %rd78;
ld.global.f64 %fd93, [%rd79];
ld.global.f64 %fd94, [%rd76];
sub.f64 %fd95, %fd94, %fd93;
add.s64 %rd80, %rd74, 128016;
add.s64 %rd81, %rd80, %rd73;
add.s64 %rd82, %rd80, %rd78;
ld.global.f64 %fd96, [%rd82];
ld.global.f64 %fd97, [%rd81];
sub.f64 %fd98, %fd97, %fd96;
mul.f64 %fd99, %fd98, %fd98;
fma.rn.f64 %fd100, %fd95, %fd95, %fd99;
sqrt.rn.f64 %fd101, %fd100;
add.s64 %rd83, %rd74, %rd77;
ld.global.f64 %fd102, [%rd83+24];
mul.f64 %fd103, %fd102, 0d3DD257B4D002790D;
ld.global.f64 %fd104, [%rd89+56];
mul.f64 %fd105, %fd103, %fd104;
fma.rn.f64 %fd106, %fd101, %fd101, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd107, %fd105, %fd106;
mul.f64 %fd108, %fd95, %fd107;
div.rn.f64 %fd109, %fd108, %fd101;
ld.global.f64 %fd110, [%rd3];
add.f64 %fd111, %fd109, %fd110;
st.global.f64 [%rd3], %fd111;
mul.f64 %fd112, %fd98, %fd107;
div.rn.f64 %fd113, %fd112, %fd101;
ld.global.f64 %fd114, [%rd4];
add.f64 %fd115, %fd113, %fd114;
st.global.f64 [%rd4], %fd115;

BB8_12:
add.s64 %rd90, %rd90, -5;
add.s64 %rd89, %rd89, 40;
add.s64 %rd88, %rd88, 5;
add.s64 %rd87, %rd87, 5;
add.s64 %rd86, %rd86, 5;
add.s64 %rd85, %rd85, 5;
add.s64 %rd84, %rd84, 5;
setp.ne.s64	%p7, %rd90, 0;
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
.reg .f64 %fd<27>;
.reg .b64 %rd<9>;


ld.param.f64 %fd4, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEdEUldE_JdEEEvT_DpT0__param_1];
mov.u32 %r1, %tid.x;
mov.u32 %r2, %ntid.x;
mov.u32 %r3, %ctaid.x;
mad.lo.s32 %r4, %r2, %r3, %r1;
cvt.s64.s32	%rd1, %r4;
ld.global.u64 %rd5, [__Bodydata_buffer];
setp.ge.u64	%p1, %rd1, %rd5;
@%p1 bra BB9_5;

shl.b64 %rd6, %rd1, 3;
mov.u64 %rd7, __Bodydata_buffer;
add.s64 %rd8, %rd7, %rd6;
ld.global.f64 %fd5, [%rd8+320024];
mul.f64 %fd6, %fd5, %fd4;
ld.global.f64 %fd7, [%rd8+24];
div.rn.f64 %fd8, %fd6, %fd7;
add.s64 %rd2, %rd8, 192024;
ld.global.f64 %fd9, [%rd8+192024];
ld.global.f64 %fd10, [%rd8+384024];
ld.global.f64 %fd11, [%rd8+256024];
add.f64 %fd12, %fd8, %fd9;
ld.global.f64 %fd13, [%rd8+64024];
ld.global.f64 %fd14, [%rd8+128024];
st.global.f64 [%rd8+192024], %fd12;
mul.f64 %fd15, %fd10, %fd4;
ld.global.f64 %fd16, [%rd8+24];
div.rn.f64 %fd17, %fd15, %fd16;
add.s64 %rd3, %rd8, 256024;
add.f64 %fd18, %fd17, %fd11;
st.global.f64 [%rd8+256024], %fd18;
fma.rn.f64 %fd19, %fd12, %fd4, %fd13;
st.global.f64 [%rd8+64024], %fd19;
ld.global.f64 %fd20, [%rd8+256024];
add.s64 %rd4, %rd8, 128024;
fma.rn.f64 %fd26, %fd20, %fd4, %fd14;
ld.global.f64 %fd21, [%rd8+64024];
st.global.f64 [%rd8+128024], %fd26;
setp.lt.f64	%p2, %fd21, 0dBFF0000000000000;
setp.gt.f64	%p3, %fd21, 0d3FF0000000000000;
or.pred %p4, %p2, %p3;
@!%p4 bra BB9_3;
bra.uni BB9_2;

BB9_2:
ld.global.f64 %fd22, [%rd2];
neg.f64 %fd23, %fd22;
st.global.f64 [%rd2], %fd23;
ld.global.f64 %fd26, [%rd4];

BB9_3:
setp.gt.f64	%p5, %fd26, 0d3FF0000000000000;
setp.lt.f64	%p6, %fd26, 0dBFF0000000000000;
or.pred %p7, %p6, %p5;
@!%p7 bra BB9_5;
bra.uni BB9_4;

BB9_4:
ld.global.f64 %fd24, [%rd3];
neg.f64 %fd25, %fd24;
st.global.f64 [%rd3], %fd25;

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
.reg .f64 %fd<92>;
.reg .b64 %rd<12>;


ld.param.f64 %fd1, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEdEUldE_JdEEEvT_DpT0__param_1];
mov.u32 %r3, %tid.x;
mov.u32 %r4, %ntid.x;
mov.u32 %r5, %ctaid.x;
mad.lo.s32 %r6, %r4, %r5, %r3;
cvt.s64.s32	%rd1, %r6;
ld.global.u64 %rd9, [__Bodydata_buffer];
setp.ge.u64	%p1, %rd1, %rd9;
@%p1 bra BB10_3;

shl.b64 %rd10, %rd1, 3;
mov.u64 %rd11, __Bodydata_buffer;
add.s64 %rd3, %rd11, %rd10;
add.s64 %rd2, %rd3, 320024;
add.s64 %rd4, %rd3, 192024;
add.s64 %rd5, %rd3, 384024;
add.s64 %rd6, %rd3, 256024;
add.s64 %rd7, %rd3, 64024;
add.s64 %rd8, %rd3, 128024;
mov.u32 %r8, -1000;

BB10_2:
ld.global.f64 %fd2, [%rd2];
mul.f64 %fd3, %fd2, %fd1;
ld.global.f64 %fd4, [%rd3+24];
div.rn.f64 %fd5, %fd3, %fd4;
ld.global.f64 %fd6, [%rd4];
add.f64 %fd7, %fd5, %fd6;
st.global.f64 [%rd4], %fd7;
ld.global.f64 %fd8, [%rd5];
mul.f64 %fd9, %fd8, %fd1;
ld.global.f64 %fd10, [%rd3+24];
div.rn.f64 %fd11, %fd9, %fd10;
ld.global.f64 %fd12, [%rd6];
add.f64 %fd13, %fd11, %fd12;
st.global.f64 [%rd6], %fd13;
ld.global.f64 %fd14, [%rd4];
ld.global.f64 %fd15, [%rd7];
fma.rn.f64 %fd16, %fd14, %fd1, %fd15;
st.global.f64 [%rd7], %fd16;
ld.global.f64 %fd17, [%rd6];
ld.global.f64 %fd18, [%rd8];
fma.rn.f64 %fd19, %fd17, %fd1, %fd18;
st.global.f64 [%rd8], %fd19;
ld.global.f64 %fd20, [%rd2];
mul.f64 %fd21, %fd20, %fd1;
ld.global.f64 %fd22, [%rd3+24];
div.rn.f64 %fd23, %fd21, %fd22;
ld.global.f64 %fd24, [%rd4];
add.f64 %fd25, %fd23, %fd24;
st.global.f64 [%rd4], %fd25;
ld.global.f64 %fd26, [%rd5];
mul.f64 %fd27, %fd26, %fd1;
ld.global.f64 %fd28, [%rd3+24];
div.rn.f64 %fd29, %fd27, %fd28;
ld.global.f64 %fd30, [%rd6];
add.f64 %fd31, %fd29, %fd30;
st.global.f64 [%rd6], %fd31;
ld.global.f64 %fd32, [%rd4];
ld.global.f64 %fd33, [%rd7];
fma.rn.f64 %fd34, %fd32, %fd1, %fd33;
st.global.f64 [%rd7], %fd34;
ld.global.f64 %fd35, [%rd6];
ld.global.f64 %fd36, [%rd8];
fma.rn.f64 %fd37, %fd35, %fd1, %fd36;
st.global.f64 [%rd8], %fd37;
ld.global.f64 %fd38, [%rd2];
mul.f64 %fd39, %fd38, %fd1;
ld.global.f64 %fd40, [%rd3+24];
div.rn.f64 %fd41, %fd39, %fd40;
ld.global.f64 %fd42, [%rd4];
add.f64 %fd43, %fd41, %fd42;
st.global.f64 [%rd4], %fd43;
ld.global.f64 %fd44, [%rd5];
mul.f64 %fd45, %fd44, %fd1;
ld.global.f64 %fd46, [%rd3+24];
div.rn.f64 %fd47, %fd45, %fd46;
ld.global.f64 %fd48, [%rd6];
add.f64 %fd49, %fd47, %fd48;
st.global.f64 [%rd6], %fd49;
ld.global.f64 %fd50, [%rd4];
ld.global.f64 %fd51, [%rd7];
fma.rn.f64 %fd52, %fd50, %fd1, %fd51;
st.global.f64 [%rd7], %fd52;
ld.global.f64 %fd53, [%rd6];
ld.global.f64 %fd54, [%rd8];
fma.rn.f64 %fd55, %fd53, %fd1, %fd54;
st.global.f64 [%rd8], %fd55;
ld.global.f64 %fd56, [%rd2];
mul.f64 %fd57, %fd56, %fd1;
ld.global.f64 %fd58, [%rd3+24];
div.rn.f64 %fd59, %fd57, %fd58;
ld.global.f64 %fd60, [%rd4];
add.f64 %fd61, %fd59, %fd60;
st.global.f64 [%rd4], %fd61;
ld.global.f64 %fd62, [%rd5];
mul.f64 %fd63, %fd62, %fd1;
ld.global.f64 %fd64, [%rd3+24];
div.rn.f64 %fd65, %fd63, %fd64;
ld.global.f64 %fd66, [%rd6];
add.f64 %fd67, %fd65, %fd66;
st.global.f64 [%rd6], %fd67;
ld.global.f64 %fd68, [%rd4];
ld.global.f64 %fd69, [%rd7];
fma.rn.f64 %fd70, %fd68, %fd1, %fd69;
st.global.f64 [%rd7], %fd70;
ld.global.f64 %fd71, [%rd6];
ld.global.f64 %fd72, [%rd8];
fma.rn.f64 %fd73, %fd71, %fd1, %fd72;
st.global.f64 [%rd8], %fd73;
ld.global.f64 %fd74, [%rd2];
mul.f64 %fd75, %fd74, %fd1;
ld.global.f64 %fd76, [%rd3+24];
div.rn.f64 %fd77, %fd75, %fd76;
ld.global.f64 %fd78, [%rd4];
add.f64 %fd79, %fd77, %fd78;
st.global.f64 [%rd4], %fd79;
ld.global.f64 %fd80, [%rd5];
mul.f64 %fd81, %fd80, %fd1;
ld.global.f64 %fd82, [%rd3+24];
div.rn.f64 %fd83, %fd81, %fd82;
ld.global.f64 %fd84, [%rd6];
add.f64 %fd85, %fd83, %fd84;
st.global.f64 [%rd6], %fd85;
ld.global.f64 %fd86, [%rd4];
ld.global.f64 %fd87, [%rd7];
fma.rn.f64 %fd88, %fd86, %fd1, %fd87;
st.global.f64 [%rd7], %fd88;
ld.global.f64 %fd89, [%rd6];
ld.global.f64 %fd90, [%rd8];
fma.rn.f64 %fd91, %fd89, %fd1, %fd90;
st.global.f64 [%rd8], %fd91;
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


