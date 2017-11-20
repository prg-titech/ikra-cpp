
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
  REG:34 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:36 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEdEUldE_JdEEEvT_DpT0_:
  REG:27 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:36 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_22add_all_forces_to_thisEvEEE4callEvEUlvE_JEEEvT_DpT0_:
  REG:43 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:321 CONSTANT[2]:64 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_11reset_forceEvEEE4callEvEUlvE_JEEEvT_DpT0_:
  REG:8 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:321 TEXTURE:0 SURFACE:0 SAMPLER:0
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
        /*00f8*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
                                                                                              /* 0x229252329283f237 */
        /*0108*/                   SSY 0x1a8;                                                 /* 0x6000000260001c07 */
        /*0110*/                   ISETP.NE.AND P0, PT, R0, RZ, PT;                           /* 0x1a8e0000fc01dc23 */
        /*0118*/                   MUFU.RCP64H R11, R7;                                       /* 0xc80000001872dc00 */
        /*0120*/                   DFMA R8, -R6, R10, c[0x2][0x0];                            /* 0x2014880000621e01 */
        /*0128*/                   DFMA R12, R8, R8, R8;                                      /* 0x2010000020831c01 */
        /*0130*/                   DMUL R8, R4, c[0x0][0x148];                                /* 0x5000400520421c01 */
        /*0138*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
                                                                                              /* 0x22b2c28282029297 */
        /*0148*/                   DMUL R4, R8, R10;                                          /* 0x5000000028811c01 */
        /*0150*/                   DFMA R14, -R6, R4, R8;                                     /* 0x2010000010639e01 */
        /*0158*/                   DFMA R10, R10, R14, R4;                                    /* 0x2008000038a29c01 */
        /*0160*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*0168*/                   FFMA R4, RZ, R7, R11;                                      /* 0x301600001ff11c00 */
        /*0170*/                   FSETP.GT.AND P1, PT, |R4|, c[0x2][0xc], PT;                /* 0x220e48003043dc80 */
        /*0178*/               @P1 BRA CC.NEU, 0x1a0;                                         /* 0x40000000800005a7 */
                                                                                              /* 0x22323042b2b2b047 */
        /*0188*/                   MOV R4, R6;                                                /* 0x2800000018011de4 */
        /*0190*/                   MOV R5, R7;                                                /* 0x280000001c015de4 */
        /*0198*/                   CAL 0xa78;                                                 /* 0x5000002360011c07 */
        /*01a0*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*01a8*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
        /*01b0*/                   MOV32I R8, 0x1;                                            /* 0x1800000004021de2 */
        /*01b8*/                   LD.E.64 R6, [R2+0x5dc18];                                  /* 0x8400177060219ca5 */
                                                                                              /* 0x2232929283f23047 */
        /*01c8*/                   LD.E.64 R24, [R2+0x2ee18];                                 /* 0x84000bb860261ca5 */
        /*01d0*/                   SSY 0x280;                                                 /* 0x60000002a0001c07 */
        /*01d8*/                   LD.E.64 R22, [R2+0x3e818];                                 /* 0x84000fa060259ca5 */
        /*01e0*/                   MUFU.RCP64H R9, R5;                                        /* 0xc800000018525c00 */
        /*01e8*/                   DFMA R12, -R4, R8, c[0x2][0x0];                            /* 0x2010880000431e01 */
        /*01f0*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*01f8*/                   DFMA R12, R8, R12, R8;                                     /* 0x2010000030831c01 */
                                                                                              /* 0x2200429232929297 */
        /*0208*/                   DMUL R8, R6, c[0x0][0x148];                                /* 0x5000400520621c01 */
        /*0210*/                   DMUL R6, R8, R12;                                          /* 0x5000000030819c01 */
        /*0218*/                   DFMA R16, -R4, R6, R8;                                     /* 0x2010000018441e01 */
        /*0220*/                   DFMA R16, R12, R16, R6;                                    /* 0x200c000040c41c01 */
        /*0228*/                   DADD R24, R10, R24;                                        /* 0x4800000060a61c01 */
        /*0230*/                   FFMA R6, RZ, R5, R17;                                      /* 0x3022000017f19c00 */
        /*0238*/                   ST.E.64 [R2+0x2ee18], R24;                                 /* 0x94000bb860261ca5 */
                                                                                              /* 0x22b2b042b2b2c277 */
        /*0248*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*0250*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003063dc80 */
        /*0258*/               @P1 BRA CC.NEU, 0x278;                                         /* 0x40000000600005a7 */
        /*0260*/                   CAL 0xa78;                                                 /* 0x5000002040011c07 */
        /*0268*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
        /*0270*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*0278*/                   NOP.S;                                                     /* 0x4000000000001df4 */
                                                                                              /* 0x22e2304230423047 */
        /*0288*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
        /*0290*/                   DADD R16, R22, R16;                                        /* 0x4800000041641c01 */
        /*0298*/                   LD.E.64 R12, [R2+0x4e218];                                 /* 0x8400138860231ca5 */
        /*02a0*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*02a8*/                   LD.E.64 R6, [R2+0xfa18];                                   /* 0x840003e860219ca5 */
        /*02b0*/                   SSY 0x3a0;                                                 /* 0x60000003a0001c07 */
        /*02b8*/                   LD.E.64 R8, [R2+0x1f418];                                  /* 0x840007d060221ca5 */
                                                                                              /* 0x229232929283f2e7 */
        /*02c8*/                   ST.E.64 [R2+0x3e818], R16;                                 /* 0x94000fa060241ca5 */
        /*02d0*/                   LD.E.64 R10, [R2+0x3e818];                                 /* 0x84000fa060229ca5 */
        /*02d8*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*02e0*/                   DFMA R18, -R4, R14, c[0x2][0x0];                           /* 0x201c880000449e01 */
        /*02e8*/                   DFMA R18, R18, R18, R18;                                   /* 0x2024000049249c01 */
        /*02f0*/                   DFMA R18, R14, R18, R14;                                   /* 0x201c000048e49c01 */
        /*02f8*/                   DMUL R14, R12, c[0x0][0x148];                              /* 0x5000400520c39c01 */
                                                                                              /* 0x2200425232329297 */
        /*0308*/                   DMUL R32, R14, R18;                                        /* 0x5000000048e81c01 */
        /*0310*/                   DFMA R12, -R4, R32, R14;                                   /* 0x201c000080431e01 */
        /*0318*/                   DFMA R18, R18, R12, R32;                                   /* 0x2040000031249c01 */
        /*0320*/                   DFMA R6, R24, c[0x0][0x148], R6;                           /* 0x200c400521819c01 */
        /*0328*/                   DFMA R8, R10, c[0x0][0x148], R8;                           /* 0x2010400520a21c01 */
        /*0330*/                   FFMA R10, RZ, R5, R19;                                     /* 0x3026000017f29c00 */
        /*0338*/                   ST.E.64 [R2+0xfa18], R6;                                   /* 0x940003e860219ca5 */
                                                                                              /* 0x22b2b042b2c042d7 */
        /*0348*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
        /*0350*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*0358*/                   ST.E.64 [R2+0x1f418], R8;                                  /* 0x940007d060221ca5 */
        /*0360*/               @P1 BRA CC.NEU, 0x398;                                         /* 0x40000000c00005a7 */
        /*0368*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0370*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
        /*0378*/                   CAL 0xa78;                                                 /* 0x5000001be0011c07 */
                                                                                              /* 0x2230423042b2b047 */
        /*0388*/                   MOV R18, R10;                                              /* 0x2800000028049de4 */
        /*0390*/                   MOV R19, R11;                                              /* 0x280000002c04dde4 */
        /*0398*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*03a0*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
        /*03a8*/                   MOV32I R8, 0x1;                                            /* 0x1800000004021de2 */
        /*03b0*/                   LD.E.64 R6, [R2+0x5dc18];                                  /* 0x8400177060219ca5 */
        /*03b8*/                   SSY 0x468;                                                 /* 0x60000002a0001c07 */
                                                                                              /* 0x229232929283f237 */
        /*03c8*/                   LD.E.64 R24, [R2+0x2ee18];                                 /* 0x84000bb860261ca5 */
        /*03d0*/                   LD.E.64 R22, [R2+0x3e818];                                 /* 0x84000fa060259ca5 */
        /*03d8*/                   MUFU.RCP64H R9, R5;                                        /* 0xc800000018525c00 */
        /*03e0*/                   DFMA R10, -R4, R8, c[0x2][0x0];                            /* 0x2010880000429e01 */
        /*03e8*/                   DFMA R10, R10, R10, R10;                                   /* 0x2014000028a29c01 */
        /*03f0*/                   DFMA R10, R8, R10, R8;                                     /* 0x2010000028829c01 */
        /*03f8*/                   DMUL R8, R6, c[0x0][0x148];                                /* 0x5000400520621c01 */
                                                                                              /* 0x2272004292329297 */
        /*0408*/                   DMUL R6, R8, R10;                                          /* 0x5000000028819c01 */
        /*0410*/                   DFMA R12, -R4, R6, R8;                                     /* 0x2010000018431e01 */
        /*0418*/                   DFMA R10, R10, R12, R6;                                    /* 0x200c000030a29c01 */
        /*0420*/                   DADD R24, R18, R24;                                        /* 0x4800000061261c01 */
        /*0428*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*0430*/                   ST.E.64 [R2+0x2ee18], R24;                                 /* 0x94000bb860261ca5 */
        /*0438*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
                                                                                              /* 0x22323042b2b2b2c7 */
        /*0448*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003063dc80 */
        /*0450*/               @P1 BRA CC.NEU, 0x460;                                         /* 0x40000000200005a7 */
        /*0458*/                   CAL 0xa78;                                                 /* 0x5000001860011c07 */
        /*0460*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0468*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
        /*0470*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*0478*/                   LD.E.64 R12, [R2+0x4e218];                                 /* 0x8400138860231ca5 */
                                                                                              /* 0x2283f2e2e0423047 */
        /*0488*/                   LD.E.64 R6, [R2+0xfa18];                                   /* 0x840003e860219ca5 */
        /*0490*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0498*/                   LD.E.64 R8, [R2+0x1f418];                                  /* 0x840007d060221ca5 */
        /*04a0*/                   SSY 0x580;                                                 /* 0x6000000360001c07 */
        /*04a8*/                   ST.E.64 [R2+0x3e818], R22;                                 /* 0x94000fa060259ca5 */
        /*04b0*/                   LD.E.64 R10, [R2+0x3e818];                                 /* 0x84000fa060229ca5 */
        /*04b8*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
                                                                                              /* 0x2232929292329297 */
        /*04c8*/                   DFMA R16, -R4, R14, c[0x2][0x0];                           /* 0x201c880000441e01 */
        /*04d0*/                   DFMA R16, R16, R16, R16;                                   /* 0x2020000041041c01 */
        /*04d8*/                   DFMA R16, R14, R16, R14;                                   /* 0x201c000040e41c01 */
        /*04e0*/                   DMUL R14, R12, c[0x0][0x148];                              /* 0x5000400520c39c01 */
        /*04e8*/                   DMUL R18, R14, R16;                                        /* 0x5000000040e49c01 */
        /*04f0*/                   DFMA R12, -R4, R18, R14;                                   /* 0x201c000048431e01 */
        /*04f8*/                   DFMA R16, R16, R12, R18;                                   /* 0x2024000031041c01 */
                                                                                              /* 0x22c042d200425237 */
        /*0508*/                   DFMA R6, R24, c[0x0][0x148], R6;                           /* 0x200c400521819c01 */
        /*0510*/                   DFMA R8, R10, c[0x0][0x148], R8;                           /* 0x2010400520a21c01 */
        /*0518*/                   FFMA R10, RZ, R5, R17;                                     /* 0x3022000017f29c00 */
        /*0520*/                   ST.E.64 [R2+0xfa18], R6;                                   /* 0x940003e860219ca5 */
        /*0528*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
        /*0530*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*0538*/                   ST.E.64 [R2+0x1f418], R8;                                  /* 0x940007d060221ca5 */
                                                                                              /* 0x22b2b042b2b042b7 */
        /*0548*/               @P1 BRA CC.NEU, 0x578;                                         /* 0x40000000a00005a7 */
        /*0550*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0558*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
        /*0560*/                   CAL 0xa78;                                                 /* 0x5000001440011c07 */
        /*0568*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
        /*0570*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*0578*/                   NOP.S;                                                     /* 0x4000000000001df4 */
                                                                                              /* 0x2283f23230423047 */
        /*0588*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
        /*0590*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*0598*/                   LD.E.64 R8, [R2+0x5dc18];                                  /* 0x8400177060221ca5 */
        /*05a0*/                   SSY 0x650;                                                 /* 0x60000002a0001c07 */
        /*05a8*/                   LD.E.64 R6, [R2+0x2ee18];                                  /* 0x84000bb860219ca5 */
        /*05b0*/                   LD.E.64 R22, [R2+0x3e818];                                 /* 0x84000fa060259ca5 */
        /*05b8*/                   MUFU.RCP64H R11, R5;                                       /* 0xc80000001852dc00 */
                                                                                              /* 0x2232929292523297 */
        /*05c8*/                   DFMA R12, -R4, R10, c[0x2][0x0];                           /* 0x2014880000431e01 */
        /*05d0*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*05d8*/                   DMUL R8, R8, c[0x0][0x148];                                /* 0x5000400520821c01 */
        /*05e0*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*05e8*/                   DMUL R14, R8, R10;                                         /* 0x5000000028839c01 */
        /*05f0*/                   DFMA R12, -R4, R14, R8;                                    /* 0x2010000038431e01 */
        /*05f8*/                   DFMA R10, R10, R12, R14;                                   /* 0x201c000030a29c01 */
                                                                                              /* 0x22b2b2c272004297 */
        /*0608*/                   DADD R24, R16, R6;                                         /* 0x4800000019061c01 */
        /*0610*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*0618*/                   ST.E.64 [R2+0x2ee18], R24;                                 /* 0x94000bb860261ca5 */
        /*0620*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*0628*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003063dc80 */
        /*0630*/               @P1 BRA CC.NEU, 0x640;                                         /* 0x40000000200005a7 */
        /*0638*/                   CAL 0xa78;                                                 /* 0x50000010e0011c07 */
                                                                                              /* 0x22304230423042b7 */
        /*0648*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0650*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
        /*0658*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*0660*/                   LD.E.64 R12, [R2+0x4e218];                                 /* 0x8400138860231ca5 */
        /*0668*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0670*/                   LD.E.64 R6, [R2+0xfa18];                                   /* 0x840003e860219ca5 */
        /*0678*/                   SSY 0x768;                                                 /* 0x60000003a0001c07 */
                                                                                              /* 0x2232929283f2e2e7 */
        /*0688*/                   LD.E.64 R8, [R2+0x1f418];                                  /* 0x840007d060221ca5 */
        /*0690*/                   ST.E.64 [R2+0x3e818], R22;                                 /* 0x94000fa060259ca5 */
        /*0698*/                   LD.E.64 R10, [R2+0x3e818];                                 /* 0x84000fa060229ca5 */
        /*06a0*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*06a8*/                   DFMA R16, -R4, R14, c[0x2][0x0];                           /* 0x201c880000441e01 */
        /*06b0*/                   DFMA R16, R16, R16, R16;                                   /* 0x2020000041041c01 */
        /*06b8*/                   DFMA R16, R14, R16, R14;                                   /* 0x201c000040e41c01 */
                                                                                              /* 0x2232123232929297 */
        /*06c8*/                   DMUL R14, R12, c[0x0][0x148];                              /* 0x5000400520c39c01 */
        /*06d0*/                   DMUL R18, R14, R16;                                        /* 0x5000000040e49c01 */
        /*06d8*/                   DFMA R12, -R4, R18, R14;                                   /* 0x201c000048431e01 */
        /*06e0*/                   DFMA R16, R16, R12, R18;                                   /* 0x2024000031041c01 */
        /*06e8*/                   DFMA R6, R24, c[0x0][0x148], R6;                           /* 0x200c400521819c01 */
        /*06f0*/                   DFMA R8, R10, c[0x0][0x148], R8;                           /* 0x2010400520a21c01 */
        /*06f8*/                   FFMA R10, RZ, R5, R17;                                     /* 0x3022000017f29c00 */
                                                                                              /* 0x22b042b042c24047 */
        /*0708*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
        /*0710*/                   ST.E.64 [R2+0xfa18], R6;                                   /* 0x940003e860219ca5 */
        /*0718*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
        /*0720*/                   ST.E.64 [R2+0x1f418], R8;                                  /* 0x940007d060221ca5 */
        /*0728*/               @P1 BRA CC.NEU, 0x760;                                         /* 0x40000000c00005a7 */
        /*0730*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0738*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
                                                                                              /* 0x22323042b2b042b7 */
        /*0748*/                   CAL 0xa78;                                                 /* 0x5000000ca0011c07 */
        /*0750*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
        /*0758*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
        /*0760*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0768*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
        /*0770*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*0778*/                   LD.E.64 R8, [R2+0x5dc18];                                  /* 0x8400177060221ca5 */
                                                                                              /* 0x2252329283f23047 */
        /*0788*/                   LD.E.64 R6, [R2+0x2ee18];                                  /* 0x84000bb860219ca5 */
        /*0790*/                   SSY 0x830;                                                 /* 0x6000000260001c07 */
        /*0798*/                   LD.E.64 R22, [R2+0x3e818];                                 /* 0x84000fa060259ca5 */
        /*07a0*/                   MUFU.RCP64H R11, R5;                                       /* 0xc80000001852dc00 */
        /*07a8*/                   DFMA R12, -R4, R10, c[0x2][0x0];                           /* 0x2014880000431e01 */
        /*07b0*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*07b8*/                   DMUL R8, R8, c[0x0][0x148];                                /* 0x5000400520821c01 */
                                                                                              /* 0x2200429232929297 */
        /*07c8*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*07d0*/                   DMUL R14, R8, R10;                                         /* 0x5000000028839c01 */
        /*07d8*/                   DFMA R12, -R4, R14, R8;                                    /* 0x2010000038431e01 */
        /*07e0*/                   DFMA R10, R10, R12, R14;                                   /* 0x201c000030a29c01 */
        /*07e8*/                   DADD R24, R16, R6;                                         /* 0x4800000019061c01 */
        /*07f0*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*07f8*/                   ST.E.64 [R2+0x2ee18], R24;                                 /* 0x94000bb860261ca5 */
                                                                                              /* 0x223042b2b2b2c277 */
        /*0808*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*0810*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003063dc80 */
        /*0818*/               @P1 BRA CC.NEU, 0x828;                                         /* 0x40000000200005a7 */
        /*0820*/                   CAL 0xa78;                                                 /* 0x5000000940011c07 */
        /*0828*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0830*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
        /*0838*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
                                                                                              /* 0x23f2e2e230423047 */
        /*0848*/                   LD.E.64 R12, [R2+0x4e218];                                 /* 0x8400138860231ca5 */
        /*0850*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0858*/                   LD.E.64 R6, [R2+0xfa18];                                   /* 0x840003e860219ca5 */
        /*0860*/                   SSY 0x950;                                                 /* 0x60000003a0001c07 */
        /*0868*/                   LD.E.64 R8, [R2+0x1f418];                                  /* 0x840007d060221ca5 */
        /*0870*/                   ST.E.64 [R2+0x3e818], R22;                                 /* 0x94000fa060259ca5 */
        /*0878*/                   LD.E.64 R10, [R2+0x3e818];                                 /* 0x84000fa060229ca5 */
                                                                                              /* 0x2292929232929287 */
        /*0888*/                   MUFU.RCP64H R15, R5;                                       /* 0xc80000001853dc00 */
        /*0890*/                   DFMA R16, -R4, R14, c[0x2][0x0];                           /* 0x201c880000441e01 */
        /*0898*/                   DFMA R16, R16, R16, R16;                                   /* 0x2020000041041c01 */
        /*08a0*/                   DFMA R16, R14, R16, R14;                                   /* 0x201c000040e41c01 */
        /*08a8*/                   DMUL R14, R12, c[0x0][0x148];                              /* 0x5000400520c39c01 */
        /*08b0*/                   DMUL R18, R14, R16;                                        /* 0x5000000040e49c01 */
        /*08b8*/                   DFMA R12, -R4, R18, R14;                                   /* 0x201c000048431e01 */
                                                                                              /* 0x22c2720042523237 */
        /*08c8*/                   DFMA R16, R16, R12, R18;                                   /* 0x2024000031041c01 */
        /*08d0*/                   DFMA R6, R24, c[0x0][0x148], R6;                           /* 0x200c400521819c01 */
        /*08d8*/                   DFMA R8, R10, c[0x0][0x148], R8;                           /* 0x2010400520a21c01 */
        /*08e0*/                   FFMA R10, RZ, R5, R17;                                     /* 0x3022000017f29c00 */
        /*08e8*/                   ST.E.64 [R2+0xfa18], R6;                                   /* 0x940003e860219ca5 */
        /*08f0*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                /* 0x160f480020ffdc80 */
        /*08f8*/                   FSETP.GT.AND P1, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a3dc80 */
                                                                                              /* 0x22b042b2b042b047 */
        /*0908*/                   ST.E.64 [R2+0x1f418], R8;                                  /* 0x940007d060221ca5 */
        /*0910*/               @P1 BRA CC.NEU, 0x940;                                         /* 0x40000000a00005a7 */
        /*0918*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0920*/                   MOV R9, R15;                                               /* 0x280000003c025de4 */
        /*0928*/                   CAL 0xa78;                                                 /* 0x5000000520011c07 */
        /*0930*/                   MOV R16, R10;                                              /* 0x2800000028041de4 */
        /*0938*/                   MOV R17, R11;                                              /* 0x280000002c045de4 */
                                                                                              /* 0x23f23230423042b7 */
        /*0948*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0950*/                   LD.E.64 R4, [R2+0x18];                                     /* 0x8400000060211ca5 */
        /*0958*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*0960*/                   LD.E.64 R8, [R2+0x5dc18];                                  /* 0x8400177060221ca5 */
        /*0968*/                   SSY 0xa18;                                                 /* 0x60000002a0001c07 */
        /*0970*/                   LD.E.64 R6, [R2+0x2ee18];                                  /* 0x84000bb860219ca5 */
        /*0978*/                   LD.E.64 R22, [R2+0x3e818];                                 /* 0x84000fa060259ca5 */
                                                                                              /* 0x2292929252329287 */
        /*0988*/                   MUFU.RCP64H R11, R5;                                       /* 0xc80000001852dc00 */
        /*0990*/                   DFMA R12, -R4, R10, c[0x2][0x0];                           /* 0x2014880000431e01 */
        /*0998*/                   DFMA R12, R12, R12, R12;                                   /* 0x2018000030c31c01 */
        /*09a0*/                   DMUL R8, R8, c[0x0][0x148];                                /* 0x5000400520821c01 */
        /*09a8*/                   DFMA R10, R10, R12, R10;                                   /* 0x2014000030a29c01 */
        /*09b0*/                   DMUL R14, R8, R10;                                         /* 0x5000000028839c01 */
        /*09b8*/                   DFMA R12, -R4, R14, R8;                                    /* 0x2010000038431e01 */
                                                                                              /* 0x22b2c27200429237 */
        /*09c8*/                   DFMA R10, R10, R12, R14;                                   /* 0x201c000030a29c01 */
        /*09d0*/                   DADD R24, R16, R6;                                         /* 0x4800000019061c01 */
        /*09d8*/                   FFMA R6, RZ, R5, R11;                                      /* 0x3016000017f19c00 */
        /*09e0*/                   ST.E.64 [R2+0x2ee18], R24;                                 /* 0x94000bb860261ca5 */
        /*09e8*/                   FSET.GTU.AND RZ.CC, |R9|, c[0x2][0x8], PT;                 /* 0x160f4800209fdc80 */
        /*09f0*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0xc], PT;                /* 0x220e48003063dc80 */
        /*09f8*/               @P1 BRA CC.NEU, 0xa10;                                         /* 0x40000000400005a7 */
                                                                                              /* 0x23f2e2e23202b2b7 */
        /*0a08*/                   CAL 0xa78;                                                 /* 0x50000001a0011c07 */
        /*0a10*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0a18*/                   DADD R22, R22, R10;                                        /* 0x4800000029659c01 */
        /*0a20*/                   LD.E.64 R4, [R2+0xfa18];                                   /* 0x840003e860211ca5 */
        /*0a28*/                   LD.E.64 R6, [R2+0x1f418];                                  /* 0x840007d060219ca5 */
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
        /*0a88*/                   MOV32I R12, 0x1ff00000;                                    /* 0x187fc00000031de2 */
        /*0a90*/                   LOP32I.AND R13, R9, 0x7f800000;                            /* 0x39fe000000935c02 */
        /*0a98*/                   ISETP.LT.U32.AND P1, PT, R6, c[0x2][0x10], PT;             /* 0x188e48004063dc03 */
        /*0aa0*/                   MOV R6, RZ;                                                /* 0x28000000fc019de4 */
        /*0aa8*/                   SEL R7, R12, c[0x2][0x14], !P1;                            /* 0x2012480050c1dc04 */
        /*0ab0*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0ab8*/                   DMUL R10, R4, R6;                                          /* 0x5000000018429c01 */
                                                                                              /* 0x2232820280428047 */
        /*0ac8*/                   ISETP.LT.U32.AND P1, PT, R13, c[0x2][0x18], PT;            /* 0x188e480060d3dc03 */
        /*0ad0*/                   MOV R18, RZ;                                               /* 0x28000000fc049de4 */
        /*0ad8*/                   MUFU.RCP64H R15, R11;                                      /* 0xc800000018b3dc00 */
        /*0ae0*/                   PBK 0xd60;                                                 /* 0x68000009e0001c07 */
        /*0ae8*/                   DFMA R20, -R10, R14, c[0x2][0x0];                          /* 0x201c880000a51e01 */
        /*0af0*/                   SEL R19, R12, c[0x2][0x14], !P1;                           /* 0x2012480050c4dc04 */
        /*0af8*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
                                                                                              /* 0x22b2d29292929257 */
        /*0b08*/                   DMUL R16, R8, R18;                                         /* 0x5000000048841c01 */
        /*0b10*/                   DFMA R14, R14, R20, R14;                                   /* 0x201c000050e39c01 */
        /*0b18*/                   DMUL R20, R16, R14;                                        /* 0x5000000039051c01 */
        /*0b20*/                   DFMA R10, -R10, R20, R16;                                  /* 0x2020000050a29e01 */
        /*0b28*/                   DFMA R10, R14, R10, R20;                                   /* 0x2028000028e29c01 */
        /*0b30*/                   DSETP.LEU.AND P1, PT, |R10|, RZ, PT;                       /* 0x1d8e0000fca3dc81 */
        /*0b38*/               @P1 BRA 0xce0;                                                 /* 0x40000006800005e7 */
                                                                                              /* 0x223252328202b207 */
        /*0b48*/                   ISETP.GT.U32.AND P1, PT, R13, c[0x2][0x1c], PT;            /* 0x1a0e480070d3dc03 */
        /*0b50*/                   DMUL R14, R6, R10;                                         /* 0x5000000028639c01 */
        /*0b58*/                   SEL R13, R12, c[0x2][0x14], !P1;                           /* 0x2012480050c35c04 */
        /*0b60*/                   MOV R12, RZ;                                               /* 0x28000000fc031de4 */
        /*0b68*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0b70*/                   DMUL R10, R12, R14;                                        /* 0x5000000038c29c01 */
        /*0b78*/                   DMUL R6, R6, R16;                                          /* 0x5000000040619c01 */
                                                                                              /* 0x22b2c27202d29257 */
        /*0b88*/                   DFMA R12, R4, R10, -R8;                                    /* 0x2010000028431d01 */
        /*0b90*/                   DFMA R14, R4, R6, -R8;                                     /* 0x2010000018439d01 */
        /*0b98*/                   DSETP.GT.AND P1, PT, |R12|, |R14|, PT;                     /* 0x1a0e000038c3dcc1 */
        /*0ba0*/                   SEL R7, R7, R11, P1;                                       /* 0x200200002c71dc04 */
        /*0ba8*/                   SEL R6, R6, R10, P1;                                       /* 0x2002000028619c04 */
        /*0bb0*/                   FSETP.GTU.AND P1, PT, |R7|, 1.469367938527859385e-39, PT;  /* 0x260ec0040073dc80 */
        /*0bb8*/               @P1 BRK;                                                       /* 0xa8000000000005e7 */
                                                                                              /* 0x2282804220420047 */
        /*0bc8*/                   FSETP.LT.AND P1, PT, |R9|, 1.5046327690525280102e-36, PT;  /* 0x208ec1000093dc80 */
        /*0bd0*/                   LOP32I.AND R10, R6, 0xfffffffe;                            /* 0x3bfffffff8629c02 */
        /*0bd8*/                   MOV32I R15, 0x3ff00000;                                    /* 0x18ffc0000003dde2 */
        /*0be0*/                   MOV R11, R7;                                               /* 0x280000001c02dde4 */
        /*0be8*/                   MOV R14, RZ;                                               /* 0x28000000fc039de4 */
        /*0bf0*/                   LOP32I.OR R6, R6, 0x1;                                     /* 0x3800000004619c42 */
        /*0bf8*/                   SEL R15, R15, c[0x2][0x20], !P1;                           /* 0x2012480080f3dc04 */
                                                                                              /* 0x220282d292329237 */
        /*0c08*/                   DMUL R4, R4, R14;                                          /* 0x5000000038411c01 */
        /*0c10*/                   DMUL R8, R8, R14;                                          /* 0x5000000038821c01 */
        /*0c18*/                   DFMA R12, R10, R4, -R8;                                    /* 0x2010000010a31d01 */
        /*0c20*/                   DFMA R14, R6, R4, -R8;                                     /* 0x2010000010639d01 */
        /*0c28*/                   DSETP.GT.AND P1, PT, |R12|, |R14|, PT;                     /* 0x1a0e000038c3dcc1 */
        /*0c30*/                   SEL R11, R6, R10, P1;                                      /* 0x200200002862dc04 */
        /*0c38*/                   IADD32I R13.CC, R11, 0x1;                                  /* 0x0c00000004b35c02 */
                                                                                              /* 0x22004202b20042b7 */
        /*0c48*/                   LOP32I.AND R10, R11, 0x1;                                  /* 0x3800000004b29c02 */
        /*0c50*/                   ISETP.EQ.U32.AND P1, PT, R10, 0x1, PT;                     /* 0x190ec00004a3dc03 */
        /*0c58*/                   IADD.X R12, RZ, R7;                                        /* 0x480000001ff31c43 */
        /*0c60*/                   IADD32I R6.CC, R11, -0x1;                                  /* 0x0ffffffffcb19c02 */
        /*0c68*/                   SEL R13, R11, R13, !P1;                                    /* 0x2012000034b35c04 */
        /*0c70*/                   IADD32I.X R10, R7, -0x1;                                   /* 0x0bfffffffc729c42 */
        /*0c78*/                   SEL R12, R7, R12, !P1;                                     /* 0x2012000030731c04 */
                                                                                              /* 0x22d2923282020277 */
        /*0c88*/                   SEL R6, R6, R11, !P1;                                      /* 0x201200002c619c04 */
        /*0c90*/                   SEL R7, R10, R7, !P1;                                      /* 0x201200001ca1dc04 */
        /*0c98*/                   MOV R10, R13;                                              /* 0x2800000034029de4 */
        /*0ca0*/                   MOV R11, R12;                                              /* 0x280000003002dde4 */
        /*0ca8*/                   DFMA R10, R4, R10, -R8;                                    /* 0x2010000028429d01 */
        /*0cb0*/                   DFMA R4, R4, R6, -R8;                                      /* 0x2010000018411d01 */
        /*0cb8*/                   DSETP.GT.AND P1, PT, |R10|, |R4|, PT;                      /* 0x1a0e000010a3dcc1 */
                                                                                              /* 0x228042b2d2e04207 */
        /*0cc8*/                   SEL R6, R6, R13, P1;                                       /* 0x2002000034619c04 */
        /*0cd0*/                   SEL R7, R7, R12, P1;                                       /* 0x200200003071dc04 */
        /*0cd8*/                   BRK;                                                       /* 0xa800000000001de7 */
        /*0ce0*/                   DSETP.EQ.AND P1, PT, R10, RZ, PT;                          /* 0x190e0000fca3dc01 */
        /*0ce8*/               @P1 BRA 0xd50;                                                 /* 0x40000001800005e7 */
        /*0cf0*/                   MUFU.RCP64H R7, R5;                                        /* 0xc80000001851dc00 */
        /*0cf8*/                   MOV R6, RZ;                                                /* 0x28000000fc019de4 */
                                                                                              /* 0x220282027202d2d7 */
        /*0d08*/                   DSETP.GT.AND P2, PT, |R6|, RZ, PT;                         /* 0x1a0e0000fc65dc81 */
        /*0d10*/              @!P2 DSETP.NEU.AND P1, PT, |R4|, +INF , PT;                     /* 0x1e8edffc0043e881 */
        /*0d18*/              @!P2 SEL R4, R4, R6, P1;                                        /* 0x2002000018412804 */
        /*0d20*/              @!P2 SEL R5, R5, R7, P1;                                        /* 0x200200001c516804 */
        /*0d28*/              @!P2 MOV R6, R4;                                                /* 0x280000001001a9e4 */
        /*0d30*/              @!P2 MOV R7, R5;                                                /* 0x280000001401e9e4 */
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
        /*0350*/                   MOV32I R0, 0x1ff00000;                                      /* 0x187fc00000001de2 */
        /*0358*/                   MOV R12, RZ;                                                /* 0x28000000fc031de4 */
        /*0360*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
        /*0368*/                   SEL R13, R0, c[0x2][0x14], !P0;                             /* 0x2010480050035c04 */
        /*0370*/                   MOV R24, RZ;                                                /* 0x28000000fc061de4 */
        /*0378*/                   DMUL R16, R10, R12;                                         /* 0x5000000030a41c01 */
                                                                                               /* 0x22925232928042c7 */
        /*0388*/                   ISETP.LT.U32.AND P0, PT, R26, c[0x2][0x18], PT;             /* 0x188e480061a1dc03 */
        /*0390*/                   MUFU.RCP64H R19, R17;                                       /* 0xc80000001914dc00 */
        /*0398*/                   SEL R25, R0, c[0x2][0x14], !P0;                             /* 0x2010480050065c04 */
        /*03a0*/                   DFMA R20, -R16, R18, c[0x2][0x0];                           /* 0x2024880001051e01 */
        /*03a8*/                   DFMA R22, R20, R20, R20;                                    /* 0x2028000051459c01 */
        /*03b0*/                   DMUL R20, R14, R24;                                         /* 0x5000000060e51c01 */
        /*03b8*/                   DFMA R18, R18, R22, R18;                                    /* 0x2024000059249c01 */
                                                                                               /* 0x220042b2d2929297 */
        /*03c8*/                   DMUL R22, R20, R18;                                         /* 0x5000000049459c01 */
        /*03d0*/                   DFMA R16, -R16, R22, R20;                                   /* 0x2028000059041e01 */
        /*03d8*/                   DFMA R22, R18, R16, R22;                                    /* 0x202c000041259c01 */
        /*03e0*/                   DSETP.LEU.AND P0, PT, |R22|, RZ, PT;                        /* 0x1d8e0000fd61dc81 */
        /*03e8*/               @P0 BRA 0x5a0;                                                  /* 0x40000006c00001e7 */
        /*03f0*/                   ISETP.GT.U32.AND P0, PT, R26, c[0x2][0x1c], PT;             /* 0x1a0e480071a1dc03 */
        /*03f8*/                   DMUL R18, R12, R22;                                         /* 0x5000000058c49c01 */
                                                                                               /* 0x22925232523282b7 */
        /*0408*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*0410*/                   SEL R17, R0, c[0x2][0x14], !P0;                             /* 0x2010480050045c04 */
        /*0418*/                   DMUL R22, R22, R16;                                         /* 0x5000000041659c01 */
        /*0420*/                   DMUL R16, R16, R18;                                         /* 0x5000000049041c01 */
        /*0428*/                   DMUL R12, R12, R22;                                         /* 0x5000000058c31c01 */
        /*0430*/                   DFMA R18, R10, R16, -R14;                                   /* 0x201c000040a49d01 */
        /*0438*/                   DFMA R20, R10, R12, -R14;                                   /* 0x201c000030a51d01 */
                                                                                               /* 0x220042b2c27202d7 */
        /*0448*/                   DSETP.GT.AND P0, PT, |R18|, |R20|, PT;                      /* 0x1a0e00005121dcc1 */
        /*0450*/                   SEL R13, R13, R17, P0;                                      /* 0x2000000044d35c04 */
        /*0458*/                   SEL R12, R12, R16, P0;                                      /* 0x2000000040c31c04 */
        /*0460*/                   FSETP.GTU.AND P0, PT, |R13|, 1.469367938527859385e-39, PT;  /* 0x260ec00400d1dc80 */
        /*0468*/               @P0 BRA 0x618;                                                  /* 0x40000006a00001e7 */
        /*0470*/                   FSETP.LT.AND P0, PT, |R15|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000f1dc80 */
        /*0478*/                   LOP32I.AND R0, R12, 0xfffffffe;                             /* 0x3bfffffff8c01c02 */
                                                                                               /* 0x2202328230427047 */
        /*0488*/                   MOV32I R17, 0x3ff00000;                                     /* 0x18ffc00000045de2 */
        /*0490*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*0498*/                   LOP32I.OR R20, R12, 0x1;                                    /* 0x3800000004c51c42 */
        /*04a0*/                   MOV R12, R0;                                                /* 0x2800000000031de4 */
        /*04a8*/                   SEL R17, R17, c[0x2][0x20], !P0;                            /* 0x2010480081145c04 */
        /*04b0*/                   DMUL R10, R10, R16;                                         /* 0x5000000040a29c01 */
        /*04b8*/                   DMUL R14, R14, R16;                                         /* 0x5000000040e39c01 */
                                                                                               /* 0x220282d292328047 */
        /*04c8*/                   MOV R16, R20;                                               /* 0x2800000050041de4 */
        /*04d0*/                   MOV R17, R13;                                               /* 0x2800000034045de4 */
        /*04d8*/                   DFMA R18, R12, R10, -R14;                                   /* 0x201c000028c49d01 */
        /*04e0*/                   DFMA R16, R16, R10, -R14;                                   /* 0x201c000029041d01 */
        /*04e8*/                   DSETP.GT.AND P0, PT, |R18|, |R16|, PT;                      /* 0x1a0e00004121dcc1 */
        /*04f0*/                   SEL R20, R20, R0, P0;                                       /* 0x2000000001451c04 */
        /*04f8*/                   IADD32I R19.CC, R20, 0x1;                                   /* 0x0c0000000544dc02 */
                                                                                               /* 0x22020042c20042b7 */
        /*0508*/                   LOP32I.AND R12, R20, 0x1;                                   /* 0x3800000005431c02 */
        /*0510*/                   ISETP.EQ.U32.AND P0, PT, R12, 0x1, PT;                      /* 0x190ec00004c1dc03 */
        /*0518*/                   IADD.X R0, RZ, R13;                                         /* 0x4800000037f01c43 */
        /*0520*/                   IADD32I R16.CC, R20, -0x1;                                  /* 0x0ffffffffd441c02 */
        /*0528*/                   SEL R19, R20, R19, !P0;                                     /* 0x201000004d44dc04 */
        /*0530*/                   IADD32I.X R17, R13, -0x1;                                   /* 0x0bfffffffcd45c42 */
        /*0538*/                   SEL R0, R13, R0, !P0;                                       /* 0x2010000000d01c04 */
                                                                                               /* 0x22d2923282020267 */
        /*0548*/                   SEL R16, R16, R20, !P0;                                     /* 0x2010000051041c04 */
        /*0550*/                   SEL R17, R17, R13, !P0;                                     /* 0x2010000035145c04 */
        /*0558*/                   MOV R12, R19;                                               /* 0x280000004c031de4 */
        /*0560*/                   MOV R13, R0;                                                /* 0x2800000000035de4 */
        /*0568*/                   DFMA R12, R10, R12, -R14;                                   /* 0x201c000030a31d01 */
        /*0570*/                   DFMA R10, R10, R16, -R14;                                   /* 0x201c000040a29d01 */
        /*0578*/                   DSETP.GT.AND P0, PT, |R12|, |R10|, PT;                      /* 0x1a0e000028c1dcc1 */
                                                                                               /* 0x228042b2d2e04207 */
        /*0588*/                   SEL R12, R16, R19, P0;                                      /* 0x200000004d031c04 */
        /*0590*/                   SEL R13, R17, R0, P0;                                       /* 0x2000000001135c04 */
        /*0598*/                   BRA 0x618;                                                  /* 0x40000001e0001de7 */
        /*05a0*/                   DSETP.EQ.AND P0, PT, R22, RZ, PT;                           /* 0x190e0000fd61dc01 */
        /*05a8*/               @P0 BRA 0x610;                                                  /* 0x40000001800001e7 */
        /*05b0*/                   MUFU.RCP64H R13, R11;                                       /* 0xc800000018b35c00 */
        /*05b8*/                   MOV R12, RZ;                                                /* 0x28000000fc031de4 */
                                                                                               /* 0x220282027202d2d7 */
        /*05c8*/                   DSETP.GT.AND P1, PT, |R12|, RZ, PT;                         /* 0x1a0e0000fcc3dc81 */
        /*05d0*/              @!P1 DSETP.NEU.AND P0, PT, |R10|, +INF , PT;                     /* 0x1e8edffc00a1e481 */
        /*05d8*/              @!P1 SEL R0, R10, R12, P0;                                       /* 0x2000000030a02404 */
        /*05e0*/              @!P1 SEL R11, R11, R13, P0;                                      /* 0x2000000034b2e404 */
        /*05e8*/              @!P1 MOV R12, R0;                                                /* 0x28000000000325e4 */
        /*05f0*/              @!P1 MOV R13, R11;                                               /* 0x280000002c0365e4 */
        /*05f8*/                   DMUL R12, R14, R12;                                         /* 0x5000000030e31c01 */
                                                                                               /* 0x200002e20042b2e7 */
        /*0608*/                   BRA 0x618;                                                  /* 0x4000000020001de7 */
        /*0610*/                   DMUL R12, R14, R10;                                         /* 0x5000000028e31c01 */
        /*0618*/                   MOV R22, R12;                                               /* 0x2800000030059de4 */
        /*0620*/                   MOV R23, R13;                                               /* 0x280000003405dde4 */
        /*0628*/                   RET;                                                        /* 0x9000000000001de7 */
        /*0630*/                   BRA 0x630;                                                  /* 0x4003ffffe0001de7 */
        /*0638*/                   NOP;                                                        /* 0x4000000000001de4 */
		..........................................................................................................................................


		Function : _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_22add_all_forces_to_thisEvEEE4callEvEUlvE_JEEEvT_DpT0_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                               /* 0x22804280423042b7 */
        /*0008*/                   MOV R1, c[0x0][0x44];                                       /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R4, 0x0;                                             /* 0x1800000000011de2 */
        /*0018*/                   S2R R41, SR_TID.X;                                          /* 0x2c000000840a5c04 */
        /*0020*/                   MOV32I R5, 0x0;                                             /* 0x1800000000015de2 */
        /*0028*/                   S2R R0, SR_CTAID.X;                                         /* 0x2c00000094001c04 */
        /*0030*/                   LD.E.64 R2, [R4];                                           /* 0x8400000000409ca5 */
        /*0038*/                   IMAD R41, R0, c[0x0][0x28], R41;                            /* 0x20524000a00a5ca3 */
                                                                                               /* 0x22020042e2c2c3f7 */
        /*0048*/                   SHR R38, R41, 0x1f;                                         /* 0x5800c0007e999c23 */
        /*0050*/                   ISUB RZ.CC, R41, R2;                                        /* 0x480100000a9fdd03 */
        /*0058*/                   ISETP.GE.U32.X.AND P0, PT, R38, R3, PT;                     /* 0x1b0e00000e61dc43 */
        /*0060*/               @P0 EXIT;                                                       /* 0x80000000000001e7 */
        /*0068*/                   IADD32I R33.CC, R41, 0x1;                                   /* 0x0c00000006985c02 */
        /*0070*/                   MOV32I R39, 0x1f40;                                         /* 0x1800007d0009dde2 */
        /*0078*/                   MOV R40, RZ;                                                /* 0x28000000fc0a1de4 */
                                                                                               /* 0x2252004210421047 */
        /*0088*/                   MOV32I R12, 0x4;                                            /* 0x1800000010031de2 */
        /*0090*/                   MOV R34, RZ;                                                /* 0x28000000fc089de4 */
        /*0098*/                   MOV32I R11, 0x3;                                            /* 0x180000000c02dde2 */
        /*00a0*/                   MOV R35, RZ;                                                /* 0x28000000fc08dde4 */
        /*00a8*/                   MOV32I R10, 0x2;                                            /* 0x1800000008029de2 */
        /*00b0*/                   MOV R36, RZ;                                                /* 0x28000000fc091de4 */
        /*00b8*/                   MOV32I R0, 0x1;                                             /* 0x1800000004001de2 */
                                                                                               /* 0x2202c202c042b047 */
        /*00c8*/                   MOV R37, RZ;                                                /* 0x28000000fc095de4 */
        /*00d0*/                   IADD.X R38, RZ, R38;                                        /* 0x480000009bf99c43 */
        /*00d8*/                   ISUB RZ.CC, R0, R33;                                        /* 0x48010000840fdd03 */
        /*00e0*/                   SSY 0x568;                                                  /* 0x6000001200001c07 */
        /*00e8*/                   ISETP.EQ.X.AND P1, PT, R37, R38, PT;                        /* 0x190e00009a53dc63 */
        /*00f0*/                   IADD32I R39.CC, R39, -0x4;                                  /* 0x0ffffffff279dc02 */
        /*00f8*/                   IADD32I.X R40, R40, -0x1;                                   /* 0x0bfffffffe8a1c42 */
                                                                                               /* 0x2270420042b042d7 */
        /*0108*/                   IADD RZ.CC, -RZ, R39;                                       /* 0x480100009fffde03 */
        /*0110*/                   ISETP.NE.X.AND P0, PT, R40, RZ, PT;                         /* 0x1a8e0000fe81dc63 */
        /*0118*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0120*/                   MOV32I R2, 0x3;                                             /* 0x180000000c009de2 */
        /*0128*/                   SHL R3, R0, -0x1d;                                          /* 0x6000ffff8c00dc03 */
        /*0130*/                   MOV32I R6, 0x0;                                             /* 0x1800000000019de2 */
        /*0138*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
                                                                                               /* 0x22220042b0420047 */
        /*0148*/                   IADD32I RZ.CC, R2, -0x20;                                   /* 0x0fffffff802fdc02 */
        /*0150*/                   SHL R7, R37, 0x3;                                           /* 0x6000c0000e51dc03 */
        /*0158*/                   MOV32I R2, 0x0;                                             /* 0x1800000000009de2 */
        /*0160*/                   MOV32I R19, 0x3fe00000;                                     /* 0x18ff80000004dde2 */
        /*0168*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
        /*0170*/                   SSY 0x2e0;                                                  /* 0x60000005a0001c07 */
        /*0178*/                   IADD32I R6.CC, R6, 0x10;                                    /* 0x0c00000040619c02 */
                                                                                               /* 0x2202c232b2004297 */
        /*0188*/                   SHR.U32 R8, R0, 0x1d;                                       /* 0x5800c00074021c03 */
        /*0190*/                   IADD.X R2, RZ, R2;                                          /* 0x480000000bf09c43 */
        /*0198*/                   IMAD R13.CC, R41, 0x8, R6;                                  /* 0x200dc00022935ca3 */
        /*01a0*/               @P1 IADD R3, R7, R8;                                            /* 0x480000002070c403 */
        /*01a8*/                   IMAD.HI.X R14, R41, 0x8, R2;                                /* 0x2084c00022939ce3 */
        /*01b0*/                   IMAD.U32.U32 R6.CC, R0, 0x8, R6;                            /* 0x200dc00020019c03 */
        /*01b8*/                   IADD.X R9, R2, R3;                                          /* 0x480000000c225c43 */
                                                                                               /* 0x23f2323280428207 */
        /*01c8*/                   MOV R2, R13;                                                /* 0x2800000034009de4 */
        /*01d0*/                   MOV R3, R14;                                                /* 0x280000003800dde4 */
        /*01d8*/                   MOV R8, R6;                                                 /* 0x2800000018021de4 */
        /*01e0*/                   LD.E.64 R24, [R2+0x1f408];                                  /* 0x840007d020261ca5 */
        /*01e8*/                   LD.E.64 R14, [R8+0x1f400];                                  /* 0x840007d000839ca5 */
        /*01f0*/                   LD.E.64 R26, [R2+0xfa08];                                   /* 0x840003e820269ca5 */
        /*01f8*/                   LD.E.64 R6, [R8+0xfa00];                                    /* 0x840003e800819ca5 */
                                                                                               /* 0x2282804292923277 */
        /*0208*/                   DADD R24, -R24, R14;                                        /* 0x4800000039861e01 */
        /*0210*/                   DADD R26, -R26, R6;                                         /* 0x4800000019a69e01 */
        /*0218*/                   DMUL R6, R24, R24;                                          /* 0x5000000061819c01 */
        /*0220*/                   DFMA R6, R26, R26, R6;                                      /* 0x200c000069a19c01 */
        /*0228*/                   IADD32I R20, R7, -0x3500000;                                /* 0x0bf2c00000751c02 */
        /*0230*/                   MUFU.RSQ64H R15, R7;                                        /* 0xc80000001c73dc00 */
        /*0238*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
                                                                                               /* 0x2202820292329297 */
        /*0248*/                   DMUL R16, R14, R14;                                         /* 0x5000000038e41c01 */
        /*0250*/                   DFMA R16, -R6, R16, c[0x2][0x0];                            /* 0x2020880000641e01 */
        /*0258*/                   DFMA R8, R16, 0.375, R18;                                   /* 0x2024cff601021c01 */
        /*0260*/                   DMUL R18, R14, R16;                                         /* 0x5000000040e49c01 */
        /*0268*/                   DFMA R8, R8, R18, R14;                                      /* 0x201c000048821c01 */
        /*0270*/                   ISETP.LT.U32.AND P1, PT, R20, c[0x2][0x8], PT;              /* 0x188e48002143dc03 */
        /*0278*/                   DMUL R16, R6, R8;                                           /* 0x5000000020641c01 */
                                                                                               /* 0x2202b20282028047 */
        /*0288*/                   IADD32I R13, R9, -0x100000;                                 /* 0x0bffc00000935c02 */
        /*0290*/                   MOV R14, R8;                                                /* 0x2800000020039de4 */
        /*0298*/                   DFMA R18, -R16, R16, R6;                                    /* 0x200c000041049e01 */
        /*02a0*/                   MOV R15, R13;                                               /* 0x280000003403dde4 */
        /*02a8*/                   DFMA R28, R18, R14, R16;                                    /* 0x2020000039271c01 */
        /*02b0*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*02b8*/                   MOV R14, R18;                                               /* 0x2800000048039de4 */
                                                                                               /* 0x2250423042b2b2b7 */
        /*02c8*/                   MOV R9, R20;                                                /* 0x2800000050025de4 */
        /*02d0*/                   CAL 0x16a8;                                                 /* 0x5000004f40011c07 */
        /*02d8*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*02e0*/                   LD.E.64 R6, [R2+0x8];                                       /* 0x8400000020219ca5 */
        /*02e8*/                   DFMA R8, R28, R28, c[0x2][0x18];                            /* 0x2038880061c21c01 */
        /*02f0*/                   LD.E.64 R16, [R4+0x18];                                     /* 0x8400000060441ca5 */
        /*02f8*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
                                                                                               /* 0x229293f292928047 */
        /*0308*/                   MUFU.RCP64H R15, R9;                                        /* 0xc80000001893dc00 */
        /*0310*/                   SSY 0x3c0;                                                  /* 0x60000002a0001c07 */
        /*0318*/                   DFMA R18, -R8, R14, c[0x2][0x0];                            /* 0x201c880000849e01 */
        /*0320*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*0328*/                   DFMA R14, R14, R18, R14;                                    /* 0x201c000048e39c01 */
        /*0330*/                   DMUL R6, R6, c[0x2][0x10];                                  /* 0x5000480040619c01 */
        /*0338*/                   DMUL R20, R6, R16;                                          /* 0x5000000040651c01 */
                                                                                               /* 0x22b2c28282029297 */
        /*0348*/                   DMUL R18, R20, R14;                                         /* 0x5000000039449c01 */
        /*0350*/                   DFMA R16, -R8, R18, R20;                                    /* 0x2028000048841e01 */
        /*0358*/                   DFMA R30, R14, R16, R18;                                    /* 0x2024000040e79c01 */
        /*0360*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
        /*0368*/                   FFMA R6, RZ, R9, R31;                                       /* 0x303e000027f19c00 */
        /*0370*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
        /*0378*/               @P1 BRA CC.NEU, 0x3b8;                                          /* 0x40000000e00005a7 */
                                                                                               /* 0x22b2b042b2b20047 */
        /*0388*/                   MOV R13, R21;                                               /* 0x2800000054035de4 */
        /*0390*/                   MOV R7, R8;                                                 /* 0x280000002001dde4 */
        /*0398*/                   MOV R6, R9;                                                 /* 0x2800000024019de4 */
        /*03a0*/                   CAL 0x1320;                                                 /* 0x5000003de0011c07 */
        /*03a8*/                   MOV R30, R6;                                                /* 0x2800000018079de4 */
        /*03b0*/                   MOV R31, R7;                                                /* 0x280000001c07dde4 */
        /*03b8*/                   MUFU.RCP64H.S R7, R29;                                      /* 0xc800000019d1dc10 */
                                                                                               /* 0x2292929292928047 */
        /*03c8*/                   MOV32I R6, 0x1;                                             /* 0x1800000004019de2 */
        /*03d0*/                   DMUL R20, R26, R30;                                         /* 0x5000000079a51c01 */
        /*03d8*/                   DFMA R8, -R28, R6, c[0x2][0x0];                             /* 0x200c880001c21e01 */
        /*03e0*/                   DFMA R8, R8, R8, R8;                                        /* 0x2010000020821c01 */
        /*03e8*/                   DFMA R8, R6, R8, R6;                                        /* 0x200c000020621c01 */
        /*03f0*/                   DMUL R6, R20, R8;                                           /* 0x5000000021419c01 */
        /*03f8*/                   DFMA R18, -R28, R6, R20;                                    /* 0x2028000019c49e01 */
                                                                                               /* 0x22b2028042804207 */
        /*0408*/                   DFMA R8, R8, R18, R6;                                       /* 0x200c000048821c01 */
        /*0410*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
        /*0418*/                   MOV R7, R28;                                                /* 0x280000007001dde4 */
        /*0420*/                   FFMA R6, RZ, R29, R9;                                       /* 0x3012000077f19c00 */
        /*0428*/                   SSY 0x478;                                                  /* 0x6000000120001c07 */
        /*0430*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
        /*0438*/                   MOV R6, R29;                                                /* 0x2800000074019de4 */
                                                                                               /* 0x2232b2b042b2b2b7 */
        /*0448*/               @P1 BRA CC.NEU, 0x470;                                          /* 0x40000000800005a7 */
        /*0450*/                   MOV R13, R21;                                               /* 0x2800000054035de4 */
        /*0458*/                   CAL 0x1320;                                                 /* 0x5000003b00011c07 */
        /*0460*/                   MOV R8, R6;                                                 /* 0x2800000018021de4 */
        /*0468*/                   MOV R9, R7;                                                 /* 0x280000001c025de4 */
        /*0470*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0478*/                   LD.E.64 R6, [R2+0x4e208];                                   /* 0x8400138820219ca5 */
                                                                                               /* 0x2292929280420047 */
        /*0488*/                   LD.E.64 R26, [R2+0x5dc08];                                  /* 0x8400177020269ca5 */
        /*0490*/                   MUFU.RCP64H R15, R29;                                       /* 0xc800000019d3dc00 */
        /*0498*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*04a0*/                   DMUL R20, R24, R30;                                         /* 0x5000000079851c01 */
        /*04a8*/                   DFMA R16, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c41e01 */
        /*04b0*/                   DFMA R16, R16, R16, R16;                                    /* 0x2020000041041c01 */
        /*04b8*/                   DFMA R18, R14, R16, R14;                                    /* 0x201c000040e49c01 */
                                                                                               /* 0x228203f042029297 */
        /*04c8*/                   DMUL R14, R18, R20;                                         /* 0x5000000051239c01 */
        /*04d0*/                   DFMA R16, -R28, R14, R20;                                   /* 0x2028000039c41e01 */
        /*04d8*/                   DFMA R16, R18, R16, R14;                                    /* 0x201c000041241c01 */
        /*04e0*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
        /*04e8*/                   SSY 0x560;                                                  /* 0x60000001c0001c07 */
        /*04f0*/                   DADD R6, R8, R6;                                            /* 0x4800000018819c01 */
        /*04f8*/                   FFMA R8, RZ, R29, R17;                                      /* 0x3022000077f21c00 */
                                                                                               /* 0x22b2b20042b2c047 */
        /*0508*/                   ST.E.64 [R2+0x4e208], R6;                                   /* 0x9400138820219ca5 */
        /*0510*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*0518*/               @P1 BRA CC.NEU, 0x558;                                          /* 0x40000000e00005a7 */
        /*0520*/                   MOV R13, R21;                                               /* 0x2800000054035de4 */
        /*0528*/                   MOV R7, R28;                                                /* 0x280000007001dde4 */
        /*0530*/                   MOV R6, R29;                                                /* 0x2800000074019de4 */
        /*0538*/                   CAL 0x1320;                                                 /* 0x5000003780011c07 */
                                                                                               /* 0x22c2c042b2b2b047 */
        /*0548*/                   MOV R16, R6;                                                /* 0x2800000018041de4 */
        /*0550*/                   MOV R17, R7;                                                /* 0x280000001c045de4 */
        /*0558*/                   DADD.S R16, R26, R16;                                       /* 0x4800000041a41c11 */
        /*0560*/                   ST.E.64.S [R2+0x5dc08], R16;                                /* 0x9400177020241cb5 */
        /*0568*/                   ISUB RZ.CC, R10, R33;                                       /* 0x4801000084afdd03 */
        /*0570*/                   SSY 0x9d8;                                                  /* 0x6000001180001c07 */
        /*0578*/                   ISETP.EQ.X.AND P1, PT, R36, R38, PT;                        /* 0x190e00009a43dc63 */
                                                                                               /* 0x22b04200428042b7 */
        /*0588*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0590*/                   MOV32I R2, 0x3;                                             /* 0x180000000c009de2 */
        /*0598*/                   SHL R3, R36, 0x3;                                           /* 0x6000c0000e40dc03 */
        /*05a0*/                   IADD32I RZ.CC, R2, -0x20;                                   /* 0x0fffffff802fdc02 */
        /*05a8*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
        /*05b0*/                   MOV32I R2, 0x0;                                             /* 0x1800000000009de2 */
        /*05b8*/                   SHR.U32 R6, R10, 0x1d;                                      /* 0x5800c00074a19c03 */
                                                                                               /* 0x220042b200420047 */
        /*05c8*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
        /*05d0*/                   MOV32I R19, 0x3fe00000;                                     /* 0x18ff80000004dde2 */
        /*05d8*/                   IADD32I R7.CC, R2, 0x10;                                    /* 0x0c0000004021dc02 */
        /*05e0*/                   SSY 0x750;                                                  /* 0x60000005a0001c07 */
        /*05e8*/                   MOV32I R2, 0x0;                                             /* 0x1800000000009de2 */
        /*05f0*/                   IADD.X R13, RZ, R2;                                         /* 0x480000000bf35c43 */
        /*05f8*/                   IMAD R14.CC, R41, 0x8, R7;                                  /* 0x200fc00022939ca3 */
                                                                                               /* 0x22720272404232b7 */
        /*0608*/                   SHL R2, R10, -0x1d;                                         /* 0x6000ffff8ca09c03 */
        /*0610*/                   IMAD.HI.X R15, R41, 0x8, R13;                               /* 0x209ac0002293dce3 */
        /*0618*/                   IMAD.U32.U32 R8.CC, R10, 0x8, R7;                           /* 0x200fc00020a21c03 */
        /*0620*/               @P1 IADD R2, R3, R6;                                            /* 0x4800000018308403 */
        /*0628*/                   MOV R3, R15;                                                /* 0x280000003c00dde4 */
        /*0630*/                   IADD.X R9, R2, R13;                                         /* 0x4800000034225c43 */
        /*0638*/                   MOV R2, R14;                                                /* 0x2800000038009de4 */
                                                                                               /* 0x22923273f2323237 */
        /*0648*/                   LD.E.64 R6, [R8+0x1f400];                                   /* 0x840007d000819ca5 */
        /*0650*/                   LD.E.64 R24, [R2+0x1f408];                                  /* 0x840007d020261ca5 */
        /*0658*/                   LD.E.64 R30, [R2+0xfa08];                                   /* 0x840003e820279ca5 */
        /*0660*/                   LD.E.64 R16, [R8+0xfa00];                                   /* 0x840003e800841ca5 */
        /*0668*/                   DADD R24, -R24, R6;                                         /* 0x4800000019861e01 */
        /*0670*/                   DADD R30, -R30, R16;                                        /* 0x4800000041e79e01 */
        /*0678*/                   DMUL R16, R24, R24;                                         /* 0x5000000061841c01 */
                                                                                               /* 0x2232929282804297 */
        /*0688*/                   DFMA R6, R30, R30, R16;                                     /* 0x2020000079e19c01 */
        /*0690*/                   IADD32I R20, R7, -0x3500000;                                /* 0x0bf2c00000751c02 */
        /*0698*/                   MUFU.RSQ64H R15, R7;                                        /* 0xc80000001c73dc00 */
        /*06a0*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*06a8*/                   DMUL R16, R14, R14;                                         /* 0x5000000038e41c01 */
        /*06b0*/                   DFMA R16, -R6, R16, c[0x2][0x0];                            /* 0x2020880000641e01 */
        /*06b8*/                   DFMA R8, R16, 0.375, R18;                                   /* 0x2024cff601021c01 */
                                                                                               /* 0x2202804202820297 */
        /*06c8*/                   DMUL R18, R14, R16;                                         /* 0x5000000040e49c01 */
        /*06d0*/                   DFMA R8, R8, R18, R14;                                      /* 0x201c000048821c01 */
        /*06d8*/                   ISETP.LT.U32.AND P1, PT, R20, c[0x2][0x8], PT;              /* 0x188e48002143dc03 */
        /*06e0*/                   DMUL R16, R6, R8;                                           /* 0x5000000020641c01 */
        /*06e8*/                   IADD32I R13, R9, -0x100000;                                 /* 0x0bffc00000935c02 */
        /*06f0*/                   MOV R14, R8;                                                /* 0x2800000020039de4 */
        /*06f8*/                   DFMA R18, -R16, R16, R6;                                    /* 0x200c000041049e01 */
                                                                                               /* 0x2202b2b042b20287 */
        /*0708*/                   MOV R15, R13;                                               /* 0x280000003403dde4 */
        /*0710*/                   DFMA R26, R18, R14, R16;                                    /* 0x2020000039269c01 */
        /*0718*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0720*/                   MOV R14, R18;                                               /* 0x2800000048039de4 */
        /*0728*/                   MOV R9, R20;                                                /* 0x2800000050025de4 */
        /*0730*/                   CAL 0x16a8;                                                 /* 0x5000003dc0011c07 */
        /*0738*/                   MOV R26, R28;                                               /* 0x2800000070069de4 */
                                                                                               /* 0x22804250423042b7 */
        /*0748*/                   MOV.S R27, R29;                                             /* 0x280000007406ddf4 */
        /*0750*/                   LD.E.64 R6, [R2+0x8];                                       /* 0x8400000020219ca5 */
        /*0758*/                   DFMA R8, R26, R26, c[0x2][0x18];                            /* 0x2034880061a21c01 */
        /*0760*/                   LD.E.64 R16, [R4+0x20];                                     /* 0x8400000080441ca5 */
        /*0768*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*0770*/                   MUFU.RCP64H R15, R9;                                        /* 0xc80000001893dc00 */
        /*0778*/                   SSY 0x830;                                                  /* 0x60000002c0001c07 */
                                                                                               /* 0x2292929293f29297 */
        /*0788*/                   DFMA R18, -R8, R14, c[0x2][0x0];                            /* 0x201c880000849e01 */
        /*0790*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*0798*/                   DFMA R14, R14, R18, R14;                                    /* 0x201c000048e39c01 */
        /*07a0*/                   DMUL R6, R6, c[0x2][0x10];                                  /* 0x5000480040619c01 */
        /*07a8*/                   DMUL R20, R6, R16;                                          /* 0x5000000040651c01 */
        /*07b0*/                   DMUL R28, R20, R14;                                         /* 0x5000000039471c01 */
        /*07b8*/                   DFMA R16, -R8, R28, R20;                                    /* 0x2028000070841e01 */
                                                                                               /* 0x220042b2c2828207 */
        /*07c8*/                   DFMA R28, R14, R16, R28;                                    /* 0x2038000040e71c01 */
        /*07d0*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
        /*07d8*/                   FFMA R6, RZ, R9, R29;                                       /* 0x303a000027f19c00 */
        /*07e0*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
        /*07e8*/               @P1 BRA CC.NEU, 0x828;                                          /* 0x40000000e00005a7 */
        /*07f0*/                   MOV R13, R21;                                               /* 0x2800000054035de4 */
        /*07f8*/                   MOV R7, R8;                                                 /* 0x280000002001dde4 */
                                                                                               /* 0x228042b2b042b2b7 */
        /*0808*/                   MOV R6, R9;                                                 /* 0x2800000024019de4 */
        /*0810*/                   CAL 0x1320;                                                 /* 0x5000002c20011c07 */
        /*0818*/                   MOV R28, R6;                                                /* 0x2800000018071de4 */
        /*0820*/                   MOV R29, R7;                                                /* 0x280000001c075de4 */
        /*0828*/                   MUFU.RCP64H.S R7, R27;                                      /* 0xc800000019b1dc10 */
        /*0830*/                   MOV32I R6, 0x1;                                             /* 0x1800000004019de2 */
        /*0838*/                   DMUL R20, R30, R28;                                         /* 0x5000000071e51c01 */
                                                                                               /* 0x2282029292929297 */
        /*0848*/                   DFMA R8, -R26, R6, c[0x2][0x0];                             /* 0x200c880001a21e01 */
        /*0850*/                   DFMA R8, R8, R8, R8;                                        /* 0x2010000020821c01 */
        /*0858*/                   DFMA R6, R6, R8, R6;                                        /* 0x200c000020619c01 */
        /*0860*/                   DMUL R8, R20, R6;                                           /* 0x5000000019421c01 */
        /*0868*/                   DFMA R18, -R26, R8, R20;                                    /* 0x2028000021a49e01 */
        /*0870*/                   DFMA R6, R6, R18, R8;                                       /* 0x2010000048619c01 */
        /*0878*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
                                                                                               /* 0x2202b042b2028047 */
        /*0888*/                   FFMA R8, RZ, R27, R7;                                       /* 0x300e00006ff21c00 */
        /*0890*/                   SSY 0x8e8;                                                  /* 0x6000000140001c07 */
        /*0898*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*08a0*/                   MOV R8, R26;                                                /* 0x2800000068021de4 */
        /*08a8*/                   MOV R9, R27;                                                /* 0x280000006c025de4 */
        /*08b0*/               @P1 BRA CC.NEU, 0x8e0;                                          /* 0x40000000a00005a7 */
        /*08b8*/                   MOV R13, R21;                                               /* 0x2800000054035de4 */
                                                                                               /* 0x22023042b2b2b047 */
        /*08c8*/                   MOV R7, R8;                                                 /* 0x280000002001dde4 */
        /*08d0*/                   MOV R6, R9;                                                 /* 0x2800000024019de4 */
        /*08d8*/                   CAL 0x1320;                                                 /* 0x5000002900011c07 */
        /*08e0*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*08e8*/                   LD.E.64 R8, [R2+0x4e208];                                   /* 0x8400138820221ca5 */
        /*08f0*/                   MUFU.RCP64H R15, R27;                                       /* 0xc800000019b3dc00 */
        /*08f8*/                   LD.E.64 R30, [R2+0x5dc08];                                  /* 0x8400177020279ca5 */
                                                                                               /* 0x2292929292928047 */
        /*0908*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*0910*/                   DMUL R20, R24, R28;                                         /* 0x5000000071851c01 */
        /*0918*/                   DFMA R16, -R26, R14, c[0x2][0x0];                           /* 0x201c880001a41e01 */
        /*0920*/                   DFMA R16, R16, R16, R16;                                    /* 0x2020000041041c01 */
        /*0928*/                   DFMA R18, R14, R16, R14;                                    /* 0x201c000040e49c01 */
        /*0930*/                   DMUL R14, R18, R20;                                         /* 0x5000000051239c01 */
        /*0938*/                   DFMA R16, -R26, R14, R20;                                   /* 0x2028000039a41e01 */
                                                                                               /* 0x22c0428203f04207 */
        /*0948*/                   DFMA R28, R18, R16, R14;                                    /* 0x201c000041271c01 */
        /*0950*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
        /*0958*/                   SSY 0x9d0;                                                  /* 0x60000001c0001c07 */
        /*0960*/                   DADD R6, R6, R8;                                            /* 0x4800000020619c01 */
        /*0968*/                   FFMA R8, RZ, R27, R29;                                      /* 0x303a00006ff21c00 */
        /*0970*/                   ST.E.64 [R2+0x4e208], R6;                                   /* 0x9400138820219ca5 */
        /*0978*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
                                                                                               /* 0x22b042b2b20042b7 */
        /*0988*/               @P1 BRA CC.NEU, 0x9c0;                                          /* 0x40000000c00005a7 */
        /*0990*/                   MOV R13, R21;                                               /* 0x2800000054035de4 */
        /*0998*/                   MOV R7, R26;                                                /* 0x280000006801dde4 */
        /*09a0*/                   MOV R6, R27;                                                /* 0x280000006c019de4 */
        /*09a8*/                   CAL 0x1320;                                                 /* 0x50000025c0011c07 */
        /*09b0*/                   MOV R28, R6;                                                /* 0x2800000018071de4 */
        /*09b8*/                   MOV R29, R7;                                                /* 0x280000001c075de4 */
                                                                                               /* 0x2282b2c2c042b2b7 */
        /*09c8*/                   DADD.S R28, R30, R28;                                       /* 0x4800000071e71c11 */
        /*09d0*/                   ST.E.64.S [R2+0x5dc08], R28;                                /* 0x9400177020271cb5 */
        /*09d8*/                   ISUB RZ.CC, R11, R33;                                       /* 0x4801000084bfdd03 */
        /*09e0*/                   SSY 0xe40;                                                  /* 0x6000001160001c07 */
        /*09e8*/                   ISETP.EQ.X.AND P1, PT, R35, R38, PT;                        /* 0x190e00009a33dc63 */
        /*09f0*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*09f8*/                   MOV32I R2, 0x3;                                             /* 0x180000000c009de2 */
                                                                                               /* 0x22020042b0420047 */
        /*0a08*/                   IADD32I RZ.CC, R2, -0x20;                                   /* 0x0fffffff802fdc02 */
        /*0a10*/                   SHL R3, R35, 0x3;                                           /* 0x6000c0000e30dc03 */
        /*0a18*/                   MOV32I R2, 0x0;                                             /* 0x1800000000009de2 */
        /*0a20*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
        /*0a28*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
        /*0a30*/                   MOV32I R19, 0x3fe00000;                                     /* 0x18ff80000004dde2 */
        /*0a38*/                   IADD32I R8.CC, R2, 0x10;                                    /* 0x0c00000040221c02 */
                                                                                               /* 0x2242720042921047 */
        /*0a48*/                   MOV32I R2, 0x0;                                             /* 0x1800000000009de2 */
        /*0a50*/                   SSY 0xbb8;                                                  /* 0x6000000580001c07 */
        /*0a58*/                   SHR.U32 R6, R11, 0x1d;                                      /* 0x5800c00074b19c03 */
        /*0a60*/                   IADD.X R7, RZ, R2;                                          /* 0x480000000bf1dc43 */
        /*0a68*/                   SHL R2, R11, -0x1d;                                         /* 0x6000ffff8cb09c03 */
        /*0a70*/                   IMAD R13.CC, R41, 0x8, R8;                                  /* 0x2011c00022935ca3 */
        /*0a78*/               @P1 IADD R2, R3, R6;                                            /* 0x4800000018308403 */
                                                                                               /* 0x2232804282c04287 */
        /*0a88*/                   IMAD.HI.X R14, R41, 0x8, R7;                                /* 0x208ec00022939ce3 */
        /*0a90*/                   IMAD.U32.U32 R8.CC, R11, 0x8, R8;                           /* 0x2011c00020b21c03 */
        /*0a98*/                   MOV R3, R14;                                                /* 0x280000003800dde4 */
        /*0aa0*/                   IADD.X R9, R2, R7;                                          /* 0x480000001c225c43 */
        /*0aa8*/                   MOV R2, R13;                                                /* 0x2800000034009de4 */
        /*0ab0*/                   LD.E.64 R6, [R8+0x1f400];                                   /* 0x840007d000819ca5 */
        /*0ab8*/                   LD.E.64 R24, [R2+0x1f408];                                  /* 0x840007d020261ca5 */
                                                                                               /* 0x228292923273f237 */
        /*0ac8*/                   LD.E.64 R30, [R2+0xfa08];                                   /* 0x840003e820279ca5 */
        /*0ad0*/                   LD.E.64 R16, [R8+0xfa00];                                   /* 0x840003e800841ca5 */
        /*0ad8*/                   DADD R24, -R24, R6;                                         /* 0x4800000019861e01 */
        /*0ae0*/                   DADD R30, -R30, R16;                                        /* 0x4800000041e79e01 */
        /*0ae8*/                   DMUL R16, R24, R24;                                         /* 0x5000000061841c01 */
        /*0af0*/                   DFMA R6, R30, R30, R16;                                     /* 0x2020000079e19c01 */
        /*0af8*/                   IADD32I R20, R7, -0x3500000;                                /* 0x0bf2c00000751c02 */
                                                                                               /* 0x2202923292928047 */
        /*0b08*/                   MUFU.RSQ64H R15, R7;                                        /* 0xc80000001c73dc00 */
        /*0b10*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*0b18*/                   DMUL R16, R14, R14;                                         /* 0x5000000038e41c01 */
        /*0b20*/                   DFMA R16, -R6, R16, c[0x2][0x0];                            /* 0x2020880000641e01 */
        /*0b28*/                   DFMA R8, R16, 0.375, R18;                                   /* 0x2024cff601021c01 */
        /*0b30*/                   DMUL R18, R14, R16;                                         /* 0x5000000040e49c01 */
        /*0b38*/                   DFMA R8, R8, R18, R14;                                      /* 0x201c000048821c01 */
                                                                                               /* 0x2202820280420287 */
        /*0b48*/                   ISETP.LT.U32.AND P1, PT, R20, c[0x2][0x8], PT;              /* 0x188e48002143dc03 */
        /*0b50*/                   DMUL R16, R6, R8;                                           /* 0x5000000020641c01 */
        /*0b58*/                   IADD32I R13, R9, -0x100000;                                 /* 0x0bffc00000935c02 */
        /*0b60*/                   MOV R14, R8;                                                /* 0x2800000020039de4 */
        /*0b68*/                   DFMA R18, -R16, R16, R6;                                    /* 0x200c000041049e01 */
        /*0b70*/                   MOV R15, R13;                                               /* 0x280000003403dde4 */
        /*0b78*/                   DFMA R26, R18, R14, R16;                                    /* 0x2020000039269c01 */
                                                                                               /* 0x2232b042b2b042b7 */
        /*0b88*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0b90*/                   MOV R14, R18;                                               /* 0x2800000048039de4 */
        /*0b98*/                   MOV R9, R20;                                                /* 0x2800000050025de4 */
        /*0ba0*/                   CAL 0x16a8;                                                 /* 0x5000002c00011c07 */
        /*0ba8*/                   MOV R26, R28;                                               /* 0x2800000070069de4 */
        /*0bb0*/                   MOV.S R27, R29;                                             /* 0x280000007406ddf4 */
        /*0bb8*/                   LD.E.64 R6, [R2+0x8];                                       /* 0x8400000020219ca5 */
                                                                                               /* 0x2292928280420047 */
        /*0bc8*/                   LD.E.64 R16, [R4+0x28];                                     /* 0x84000000a0441ca5 */
        /*0bd0*/                   DFMA R8, R26, R26, c[0x2][0x18];                            /* 0x2034880061a21c01 */
        /*0bd8*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*0be0*/                   SSY 0xca0;                                                  /* 0x60000002e0001c07 */
        /*0be8*/                   MUFU.RCP64H R15, R9;                                        /* 0xc80000001893dc00 */
        /*0bf0*/                   DFMA R18, -R8, R14, c[0x2][0x0];                            /* 0x201c880000849e01 */
        /*0bf8*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
                                                                                               /* 0x22820292929293f7 */
        /*0c08*/                   DFMA R14, R14, R18, R14;                                    /* 0x201c000048e39c01 */
        /*0c10*/                   DMUL R6, R6, c[0x2][0x10];                                  /* 0x5000480040619c01 */
        /*0c18*/                   DMUL R20, R6, R16;                                          /* 0x5000000040651c01 */
        /*0c20*/                   DMUL R28, R20, R14;                                         /* 0x5000000039471c01 */
        /*0c28*/                   DFMA R16, -R8, R28, R20;                                    /* 0x2028000070841e01 */
        /*0c30*/                   DFMA R28, R14, R16, R28;                                    /* 0x2038000040e71c01 */
        /*0c38*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
                                                                                               /* 0x22b2b20042b2c287 */
        /*0c48*/                   FFMA R6, RZ, R9, R29;                                       /* 0x303a000027f19c00 */
        /*0c50*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
        /*0c58*/               @P1 BRA CC.NEU, 0xc98;                                          /* 0x40000000e00005a7 */
        /*0c60*/                   MOV R13, R21;                                               /* 0x2800000054035de4 */
        /*0c68*/                   MOV R7, R8;                                                 /* 0x280000002001dde4 */
        /*0c70*/                   MOV R6, R9;                                                 /* 0x2800000024019de4 */
        /*0c78*/                   CAL 0x1320;                                                 /* 0x5000001a80011c07 */
                                                                                               /* 0x2292928042b2b047 */
        /*0c88*/                   MOV R28, R6;                                                /* 0x2800000018071de4 */
        /*0c90*/                   MOV R29, R7;                                                /* 0x280000001c075de4 */
        /*0c98*/                   MUFU.RCP64H.S R7, R27;                                      /* 0xc800000019b1dc10 */
        /*0ca0*/                   MOV32I R6, 0x1;                                             /* 0x1800000004019de2 */
        /*0ca8*/                   DMUL R20, R30, R28;                                         /* 0x5000000071e51c01 */
        /*0cb0*/                   DFMA R8, -R26, R6, c[0x2][0x0];                             /* 0x200c880001a21e01 */
        /*0cb8*/                   DFMA R8, R8, R8, R8;                                        /* 0x2010000020821c01 */
                                                                                               /* 0x2282804202929297 */
        /*0cc8*/                   DFMA R6, R6, R8, R6;                                        /* 0x200c000020619c01 */
        /*0cd0*/                   DMUL R8, R20, R6;                                           /* 0x5000000019421c01 */
        /*0cd8*/                   DFMA R18, -R26, R8, R20;                                    /* 0x2028000021a49e01 */
        /*0ce0*/                   DFMA R6, R6, R18, R8;                                       /* 0x2010000048619c01 */
        /*0ce8*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
        /*0cf0*/                   MOV R9, R27;                                                /* 0x280000006c025de4 */
        /*0cf8*/                   FFMA R8, RZ, R27, R7;                                       /* 0x300e00006ff21c00 */
                                                                                               /* 0x22b20042b042c047 */
        /*0d08*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*0d10*/                   SSY 0xd58;                                                  /* 0x6000000100001c07 */
        /*0d18*/                   MOV R8, R26;                                                /* 0x2800000068021de4 */
        /*0d20*/               @P1 BRA CC.NEU, 0xd50;                                          /* 0x40000000a00005a7 */
        /*0d28*/                   MOV R13, R21;                                               /* 0x2800000054035de4 */
        /*0d30*/                   MOV R7, R8;                                                 /* 0x280000002001dde4 */
        /*0d38*/                   MOV R6, R9;                                                 /* 0x2800000024019de4 */
                                                                                               /* 0x228200423042b2b7 */
        /*0d48*/                   CAL 0x1320;                                                 /* 0x5000001740011c07 */
        /*0d50*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0d58*/                   LD.E.64 R8, [R2+0x4e208];                                   /* 0x8400138820221ca5 */
        /*0d60*/                   MUFU.RCP64H R15, R27;                                       /* 0xc800000019b3dc00 */
        /*0d68*/                   LD.E.64 R30, [R2+0x5dc08];                                  /* 0x8400177020279ca5 */
        /*0d70*/                   DMUL R20, R24, R28;                                         /* 0x5000000071851c01 */
        /*0d78*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
                                                                                               /* 0x2202029292929297 */
        /*0d88*/                   DFMA R16, -R26, R14, c[0x2][0x0];                           /* 0x201c880001a41e01 */
        /*0d90*/                   DFMA R16, R16, R16, R16;                                    /* 0x2020000041041c01 */
        /*0d98*/                   DFMA R18, R14, R16, R14;                                    /* 0x201c000040e49c01 */
        /*0da0*/                   DMUL R14, R18, R20;                                         /* 0x5000000051239c01 */
        /*0da8*/                   DFMA R16, -R26, R14, R20;                                   /* 0x2028000039a41e01 */
        /*0db0*/                   DFMA R28, R18, R16, R14;                                    /* 0x201c000041271c01 */
        /*0db8*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
                                                                                               /* 0x2202b2c0428203f7 */
        /*0dc8*/                   SSY 0xe38;                                                  /* 0x60000001a0001c07 */
        /*0dd0*/                   DADD R6, R6, R8;                                            /* 0x4800000020619c01 */
        /*0dd8*/                   FFMA R8, RZ, R27, R29;                                      /* 0x303a00006ff21c00 */
        /*0de0*/                   ST.E.64 [R2+0x4e208], R6;                                   /* 0x9400138820219ca5 */
        /*0de8*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*0df0*/               @P1 BRA CC.NEU, 0xe30;                                          /* 0x40000000e00005a7 */
        /*0df8*/                   MOV R13, R21;                                               /* 0x2800000054035de4 */
                                                                                               /* 0x22b2b2b042b2b047 */
        /*0e08*/                   MOV R7, R26;                                                /* 0x280000006801dde4 */
        /*0e10*/                   MOV R6, R27;                                                /* 0x280000006c019de4 */
        /*0e18*/                   CAL 0x1320;                                                 /* 0x5000001400011c07 */
        /*0e20*/                   MOV R28, R6;                                                /* 0x2800000018071de4 */
        /*0e28*/                   MOV R29, R7;                                                /* 0x280000001c075de4 */
        /*0e30*/                   DADD.S R28, R30, R28;                                       /* 0x4800000071e71c11 */
        /*0e38*/                   ST.E.64.S [R2+0x5dc08], R28;                                /* 0x9400177020271cb5 */
                                                                                               /* 0x22028042b2c2c047 */
        /*0e48*/                   ISUB RZ.CC, R12, R33;                                       /* 0x4801000084cfdd03 */
        /*0e50*/                   SSY 0x12b0;                                                 /* 0x6000001160001c07 */
        /*0e58*/                   ISETP.EQ.X.AND P1, PT, R34, R38, PT;                        /* 0x190e00009a23dc63 */
        /*0e60*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0e68*/                   MOV32I R2, 0x3;                                             /* 0x180000000c009de2 */
        /*0e70*/                   SHL R3, R34, 0x3;                                           /* 0x6000c0000e20dc03 */
        /*0e78*/                   IADD32I RZ.CC, R2, -0x20;                                   /* 0x0fffffff802fdc02 */
                                                                                               /* 0x22b200420042b047 */
        /*0e88*/                   MOV32I R2, 0x0;                                             /* 0x1800000000009de2 */
        /*0e90*/                   SHR.U32 R6, R12, 0x1d;                                      /* 0x5800c00074c19c03 */
        /*0e98*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
        /*0ea0*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
        /*0ea8*/                   IADD32I R8.CC, R2, 0x10;                                    /* 0x0c00000040221c02 */
        /*0eb0*/                   MOV32I R19, 0x3fe00000;                                     /* 0x18ff80000004dde2 */
        /*0eb8*/                   MOV32I R2, 0x0;                                             /* 0x1800000000009de2 */
                                                                                               /* 0x22c2324270420047 */
        /*0ec8*/                   IADD.X R7, RZ, R2;                                          /* 0x480000000bf1dc43 */
        /*0ed0*/                   SHL R2, R12, -0x1d;                                         /* 0x6000ffff8cc09c03 */
        /*0ed8*/                   IMAD R13.CC, R41, 0x8, R8;                                  /* 0x2011c00022935ca3 */
        /*0ee0*/                   SSY 0x1028;                                                 /* 0x6000000500001c07 */
        /*0ee8*/               @P1 IADD R2, R3, R6;                                            /* 0x4800000018308403 */
        /*0ef0*/                   IMAD.HI.X R14, R41, 0x8, R7;                                /* 0x208ec00022939ce3 */
        /*0ef8*/                   IMAD.U32.U32 R8.CC, R12, 0x8, R8;                           /* 0x2011c00020c21c03 */
                                                                                               /* 0x23f2323280428047 */
        /*0f08*/                   MOV R3, R14;                                                /* 0x280000003800dde4 */
        /*0f10*/                   IADD.X R9, R2, R7;                                          /* 0x480000001c225c43 */
        /*0f18*/                   MOV R2, R13;                                                /* 0x2800000034009de4 */
        /*0f20*/                   LD.E.64 R6, [R8+0x1f400];                                   /* 0x840007d000819ca5 */
        /*0f28*/                   LD.E.64 R24, [R2+0x1f408];                                  /* 0x840007d020261ca5 */
        /*0f30*/                   LD.E.64 R30, [R2+0xfa08];                                   /* 0x840003e820279ca5 */
        /*0f38*/                   LD.E.64 R16, [R8+0xfa00];                                   /* 0x840003e800841ca5 */
                                                                                               /* 0x2282804292923277 */
        /*0f48*/                   DADD R24, -R24, R6;                                         /* 0x4800000019861e01 */
        /*0f50*/                   DADD R30, -R30, R16;                                        /* 0x4800000041e79e01 */
        /*0f58*/                   DMUL R16, R24, R24;                                         /* 0x5000000061841c01 */
        /*0f60*/                   DFMA R6, R30, R30, R16;                                     /* 0x2020000079e19c01 */
        /*0f68*/                   IADD32I R20, R7, -0x3500000;                                /* 0x0bf2c00000751c02 */
        /*0f70*/                   MUFU.RSQ64H R15, R7;                                        /* 0xc80000001c73dc00 */
        /*0f78*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
                                                                                               /* 0x2202820292329297 */
        /*0f88*/                   DMUL R16, R14, R14;                                         /* 0x5000000038e41c01 */
        /*0f90*/                   DFMA R16, -R6, R16, c[0x2][0x0];                            /* 0x2020880000641e01 */
        /*0f98*/                   DFMA R8, R16, 0.375, R18;                                   /* 0x2024cff601021c01 */
        /*0fa0*/                   DMUL R18, R14, R16;                                         /* 0x5000000040e49c01 */
        /*0fa8*/                   DFMA R8, R8, R18, R14;                                      /* 0x201c000048821c01 */
        /*0fb0*/                   ISETP.LT.U32.AND P1, PT, R20, c[0x2][0x8], PT;              /* 0x188e48002143dc03 */
        /*0fb8*/                   DMUL R16, R6, R8;                                           /* 0x5000000020641c01 */
                                                                                               /* 0x2202b20282028047 */
        /*0fc8*/                   IADD32I R13, R9, -0x100000;                                 /* 0x0bffc00000935c02 */
        /*0fd0*/                   MOV R14, R8;                                                /* 0x2800000020039de4 */
        /*0fd8*/                   DFMA R18, -R16, R16, R6;                                    /* 0x200c000041049e01 */
        /*0fe0*/                   MOV R15, R13;                                               /* 0x280000003403dde4 */
        /*0fe8*/                   DFMA R26, R18, R14, R16;                                    /* 0x2020000039269c01 */
        /*0ff0*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0ff8*/                   MOV R14, R18;                                               /* 0x2800000048039de4 */
                                                                                               /* 0x22023042b042b2b7 */
        /*1008*/                   MOV R9, R20;                                                /* 0x2800000050025de4 */
        /*1010*/                   CAL 0x16a8;                                                 /* 0x5000001a40011c07 */
        /*1018*/                   MOV R26, R28;                                               /* 0x2800000070069de4 */
        /*1020*/                   MOV.S R27, R29;                                             /* 0x280000007406ddf4 */
        /*1028*/                   LD.E.64 R6, [R2+0x8];                                       /* 0x8400000020219ca5 */
        /*1030*/                   DFMA R8, R26, R26, c[0x2][0x18];                            /* 0x2034880061a21c01 */
        /*1038*/                   LD.E.64 R16, [R4+0x30];                                     /* 0x84000000c0441ca5 */
                                                                                               /* 0x2293f29292824047 */
        /*1048*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*1050*/                   SSY 0x1110;                                                 /* 0x60000002e0001c07 */
        /*1058*/                   MUFU.RCP64H R15, R9;                                        /* 0xc80000001893dc00 */
        /*1060*/                   DFMA R18, -R8, R14, c[0x2][0x0];                            /* 0x201c880000849e01 */
        /*1068*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*1070*/                   DFMA R14, R14, R18, R14;                                    /* 0x201c000048e39c01 */
        /*1078*/                   DMUL R6, R6, c[0x2][0x10];                                  /* 0x5000480040619c01 */
                                                                                               /* 0x22c2828202929297 */
        /*1088*/                   DMUL R20, R6, R16;                                          /* 0x5000000040651c01 */
        /*1090*/                   DMUL R28, R20, R14;                                         /* 0x5000000039471c01 */
        /*1098*/                   DFMA R16, -R8, R28, R20;                                    /* 0x2028000070841e01 */
        /*10a0*/                   DFMA R28, R14, R16, R28;                                    /* 0x2038000040e71c01 */
        /*10a8*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
        /*10b0*/                   FFMA R6, RZ, R9, R29;                                       /* 0x303a000027f19c00 */
        /*10b8*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
                                                                                               /* 0x22b042b2b20042b7 */
        /*10c8*/               @P1 BRA CC.NEU, 0x1100;                                         /* 0x40000000c00005a7 */
        /*10d0*/                   MOV R13, R21;                                               /* 0x2800000054035de4 */
        /*10d8*/                   MOV R7, R8;                                                 /* 0x280000002001dde4 */
        /*10e0*/                   MOV R6, R9;                                                 /* 0x2800000024019de4 */
        /*10e8*/                   CAL 0x1320;                                                 /* 0x50000008c0011c07 */
        /*10f0*/                   MOV R28, R6;                                                /* 0x2800000018071de4 */
        /*10f8*/                   MOV R29, R7;                                                /* 0x280000001c075de4 */
                                                                                               /* 0x22929292928042b7 */
        /*1108*/                   MUFU.RCP64H.S R7, R27;                                      /* 0xc800000019b1dc10 */
        /*1110*/                   MOV32I R6, 0x1;                                             /* 0x1800000004019de2 */
        /*1118*/                   DMUL R20, R30, R28;                                         /* 0x5000000071e51c01 */
        /*1120*/                   DFMA R8, -R26, R6, c[0x2][0x0];                             /* 0x200c880001a21e01 */
        /*1128*/                   DFMA R8, R8, R8, R8;                                        /* 0x2010000020821c01 */
        /*1130*/                   DFMA R6, R6, R8, R6;                                        /* 0x200c000020619c01 */
        /*1138*/                   DMUL R8, R20, R6;                                           /* 0x5000000019421c01 */
                                                                                               /* 0x22c2804280420297 */
        /*1148*/                   DFMA R18, -R26, R8, R20;                                    /* 0x2028000021a49e01 */
        /*1150*/                   DFMA R6, R6, R18, R8;                                       /* 0x2010000048619c01 */
        /*1158*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
        /*1160*/                   MOV R9, R27;                                                /* 0x280000006c025de4 */
        /*1168*/                   FFMA R8, RZ, R27, R7;                                       /* 0x300e00006ff21c00 */
        /*1170*/                   SSY 0x11c0;                                                 /* 0x6000000120001c07 */
        /*1178*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
                                                                                               /* 0x22b2b2b20042b047 */
        /*1188*/                   MOV R8, R26;                                                /* 0x2800000068021de4 */
        /*1190*/               @P1 BRA CC.NEU, 0x11b8;                                         /* 0x40000000800005a7 */
        /*1198*/                   MOV R13, R21;                                               /* 0x2800000054035de4 */
        /*11a0*/                   MOV R7, R8;                                                 /* 0x280000002001dde4 */
        /*11a8*/                   MOV R6, R9;                                                 /* 0x2800000024019de4 */
        /*11b0*/                   CAL 0x1320;                                                 /* 0x50000005a0011c07 */
        /*11b8*/                   NOP.S;                                                      /* 0x4000000000001df4 */
                                                                                               /* 0x2292804200423047 */
        /*11c8*/                   LD.E.64 R8, [R2+0x4e208];                                   /* 0x8400138820221ca5 */
        /*11d0*/                   MUFU.RCP64H R15, R27;                                       /* 0xc800000019b3dc00 */
        /*11d8*/                   LD.E.64 R30, [R2+0x5dc08];                                  /* 0x8400177020279ca5 */
        /*11e0*/                   DMUL R20, R24, R28;                                         /* 0x5000000071851c01 */
        /*11e8*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*11f0*/                   SSY 0x12a8;                                                 /* 0x60000002c0001c07 */
        /*11f8*/                   DFMA R16, -R26, R14, c[0x2][0x0];                           /* 0x201c880001a41e01 */
                                                                                               /* 0x2203f20292929297 */
        /*1208*/                   DFMA R16, R16, R16, R16;                                    /* 0x2020000041041c01 */
        /*1210*/                   DFMA R18, R14, R16, R14;                                    /* 0x201c000040e49c01 */
        /*1218*/                   DMUL R14, R18, R20;                                         /* 0x5000000051239c01 */
        /*1220*/                   DFMA R16, -R26, R14, R20;                                   /* 0x2028000039a41e01 */
        /*1228*/                   DFMA R28, R18, R16, R14;                                    /* 0x201c000041271c01 */
        /*1230*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
        /*1238*/                   DADD R6, R6, R8;                                            /* 0x4800000020619c01 */
                                                                                               /* 0x22b20042b2c04287 */
        /*1248*/                   FFMA R8, RZ, R27, R29;                                      /* 0x303a00006ff21c00 */
        /*1250*/                   ST.E.64 [R2+0x4e208], R6;                                   /* 0x9400138820219ca5 */
        /*1258*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*1260*/               @P1 BRA CC.NEU, 0x12a0;                                         /* 0x40000000e00005a7 */
        /*1268*/                   MOV R13, R21;                                               /* 0x2800000054035de4 */
        /*1270*/                   MOV R7, R26;                                                /* 0x280000006801dde4 */
        /*1278*/                   MOV R6, R27;                                                /* 0x280000006c019de4 */
                                                                                               /* 0x2202c2b2b2b042b7 */
        /*1288*/                   CAL 0x1320;                                                 /* 0x5000000240011c07 */
        /*1290*/                   MOV R28, R6;                                                /* 0x2800000018071de4 */
        /*1298*/                   MOV R29, R7;                                                /* 0x280000001c075de4 */
        /*12a0*/                   DADD.S R28, R30, R28;                                       /* 0x4800000071e71c11 */
        /*12a8*/                   ST.E.64.S [R2+0x5dc08], R28;                                /* 0x9400177020271cb5 */
        /*12b0*/                   IADD32I R4.CC, R4, 0x20;                                    /* 0x0c00000080411c02 */
        /*12b8*/                   IADD.X R5, RZ, R5;                                          /* 0x4800000017f15c43 */
                                                                                               /* 0x22c042c042c042c7 */
        /*12c8*/                   IADD32I R12.CC, R12, 0x4;                                   /* 0x0c00000010c31c02 */
        /*12d0*/                   IADD.X R34, RZ, R34;                                        /* 0x480000008bf89c43 */
        /*12d8*/                   IADD32I R11.CC, R11, 0x4;                                   /* 0x0c00000010b2dc02 */
        /*12e0*/                   IADD.X R35, RZ, R35;                                        /* 0x480000008ff8dc43 */
        /*12e8*/                   IADD32I R10.CC, R10, 0x4;                                   /* 0x0c00000010a29c02 */
        /*12f0*/                   IADD.X R36, RZ, R36;                                        /* 0x4800000093f91c43 */
        /*12f8*/                   IADD32I R0.CC, R0, 0x4;                                     /* 0x0c00000010001c02 */
                                                                                               /* 0x2270420042e2e047 */
        /*1308*/                   IADD.X R37, RZ, R37;                                        /* 0x4800000097f95c43 */
        /*1310*/               @P0 BRA 0xd8;                                                   /* 0x4003ffb7000001e7 */
        /*1318*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*1320*/                   LOP32I.AND R8, R6, 0x40000000;                              /* 0x3900000000621c02 */
        /*1328*/                   LOP.XOR R7, R7, R6;                                         /* 0x680000001871dc83 */
        /*1330*/                   MOV32I R42, 0x1ff00000;                                     /* 0x187fc000000a9de2 */
        /*1338*/                   PBK 0x1690;                                                 /* 0x6800000d40001c07 */
                                                                                               /* 0x228200428042c047 */
        /*1348*/                   ISETP.LT.U32.AND P1, PT, R8, c[0x2][0x24], PT;              /* 0x188e48009083dc03 */
        /*1350*/                   LOP.XOR R6, R7, R6;                                         /* 0x6800000018719c83 */
        /*1358*/                   SEL R14, R42, c[0x2][0x28], !P1;                            /* 0x20124800a2a39c04 */
        /*1360*/                   LOP.XOR R7, R7, R6;                                         /* 0x680000001871dc83 */
        /*1368*/                   MOV R15, R14;                                               /* 0x280000003803dde4 */
        /*1370*/                   LOP32I.AND R32, R13, 0x7f800000;                            /* 0x39fe000000d81c02 */
        /*1378*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
                                                                                               /* 0x2282028042804297 */
        /*1388*/                   DMUL R18, R6, R14;                                          /* 0x5000000038649c01 */
        /*1390*/                   MUFU.RCP64H R16, R19;                                       /* 0xc800000019341c00 */
        /*1398*/                   ISETP.LT.U32.AND P1, PT, R32, c[0x2][0x2c], PT;             /* 0x188e4800b203dc03 */
        /*13a0*/                   MOV R17, R16;                                               /* 0x2800000040045de4 */
        /*13a8*/                   MOV32I R16, 0x1;                                            /* 0x1800000004041de2 */
        /*13b0*/                   DFMA R8, -R18, R16, c[0x2][0x0];                            /* 0x2020880001221e01 */
        /*13b8*/                   SEL R22, R42, c[0x2][0x28], !P1;                            /* 0x20124800a2a59c04 */
                                                                                               /* 0x2292804202804207 */
        /*13c8*/                   DFMA R8, R8, R8, R8;                                        /* 0x2010000020821c01 */
        /*13d0*/                   MOV R23, R22;                                               /* 0x280000005805dde4 */
        /*13d8*/                   MOV R22, RZ;                                                /* 0x28000000fc059de4 */
        /*13e0*/                   DFMA R16, R16, R8, R16;                                     /* 0x2020000021041c01 */
        /*13e8*/                   MOV R8, R20;                                                /* 0x2800000050021de4 */
        /*13f0*/                   MOV R9, R13;                                                /* 0x2800000034025de4 */
        /*13f8*/                   DMUL R22, R8, R22;                                          /* 0x5000000058859c01 */
                                                                                               /* 0x22c042b2d2929297 */
        /*1408*/                   DMUL R20, R22, R16;                                         /* 0x5000000041651c01 */
        /*1410*/                   DFMA R18, -R18, R20, R22;                                   /* 0x202c000051249e01 */
        /*1418*/                   DFMA R20, R16, R18, R20;                                    /* 0x2028000049051c01 */
        /*1420*/                   DSETP.LEU.AND P1, PT, |R20|, RZ, PT;                        /* 0x1d8e0000fd43dc81 */
        /*1428*/               @P1 BRA 0x1610;                                                 /* 0x40000007800005e7 */
        /*1430*/                   ISETP.GT.U32.AND P1, PT, R32, c[0x2][0x30], PT;             /* 0x1a0e4800c203dc03 */
        /*1438*/                   DMUL R18, R14, R20;                                         /* 0x5000000050e49c01 */
                                                                                               /* 0x2252325232804287 */
        /*1448*/                   SEL R16, R42, c[0x2][0x28], !P1;                            /* 0x20124800a2a41c04 */
        /*1450*/                   MOV R17, R16;                                               /* 0x2800000040045de4 */
        /*1458*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*1460*/                   DMUL R20, R20, R16;                                         /* 0x5000000041451c01 */
        /*1468*/                   DMUL R16, R16, R18;                                         /* 0x5000000049041c01 */
        /*1470*/                   DMUL R14, R14, R20;                                         /* 0x5000000050e39c01 */
        /*1478*/                   DFMA R18, R6, R16, -R8;                                     /* 0x2010000040649d01 */
                                                                                               /* 0x22b042c27202d297 */
        /*1488*/                   DFMA R20, R6, R14, -R8;                                     /* 0x2010000038651d01 */
        /*1490*/                   DSETP.GT.AND P1, PT, |R18|, |R20|, PT;                      /* 0x1a0e00005123dcc1 */
        /*1498*/                   SEL R13, R15, R17, P1;                                      /* 0x2002000044f35c04 */
        /*14a0*/                   SEL R16, R14, R16, P1;                                      /* 0x2002000040e41c04 */
        /*14a8*/                   FSETP.GTU.AND P1, PT, |R13|, 1.469367938527859385e-39, PT;  /* 0x260ec00400d3dc80 */
        /*14b0*/                   MOV R17, R13;                                               /* 0x2800000034045de4 */
        /*14b8*/               @P1 BRK;                                                        /* 0xa8000000000005e7 */
                                                                                               /* 0x2282304270420047 */
        /*14c8*/                   FSETP.LT.AND P1, PT, |R9|, 1.5046327690525280102e-36, PT;   /* 0x208ec1000093dc80 */
        /*14d0*/                   LOP32I.AND R19, R16, 0xfffffffe;                            /* 0x3bfffffff904dc02 */
        /*14d8*/                   MOV32I R14, 0x3ff00000;                                     /* 0x18ffc00000039de2 */
        /*14e0*/                   MOV R17, R13;                                               /* 0x2800000034045de4 */
        /*14e8*/                   LOP32I.OR R18, R16, 0x1;                                    /* 0x3800000005049c42 */
        /*14f0*/                   MOV R16, R19;                                               /* 0x280000004c041de4 */
        /*14f8*/                   SEL R14, R14, c[0x2][0x34], !P1;                            /* 0x20124800d0e39c04 */
                                                                                               /* 0x2232804202328047 */
        /*1508*/                   MOV R15, R14;                                               /* 0x280000003803dde4 */
        /*1510*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
        /*1518*/                   DMUL R6, R6, R14;                                           /* 0x5000000038619c01 */
        /*1520*/                   DMUL R8, R8, R14;                                           /* 0x5000000038821c01 */
        /*1528*/                   MOV R14, R18;                                               /* 0x2800000048039de4 */
        /*1530*/                   MOV R15, R13;                                               /* 0x280000003403dde4 */
        /*1538*/                   DFMA R16, R16, R6, -R8;                                     /* 0x2010000019041d01 */
                                                                                               /* 0x220042c04282d297 */
        /*1548*/                   DFMA R14, R14, R6, -R8;                                     /* 0x2010000018e39d01 */
        /*1550*/                   DSETP.GT.AND P1, PT, |R16|, |R14|, PT;                      /* 0x1a0e00003903dcc1 */
        /*1558*/                   SEL R14, R18, R19, P1;                                      /* 0x200200004d239c04 */
        /*1560*/                   IADD32I R19.CC, R14, 0x1;                                   /* 0x0c00000004e4dc02 */
        /*1568*/                   LOP32I.AND R15, R14, 0x1;                                   /* 0x3800000004e3dc02 */
        /*1570*/                   IADD.X R18, RZ, R13;                                        /* 0x4800000037f49c43 */
        /*1578*/                   IADD32I R16.CC, R14, -0x1;                                  /* 0x0ffffffffce41c02 */
                                                                                               /* 0x22020262020042c7 */
        /*1588*/                   ISETP.EQ.U32.AND P1, PT, R15, 0x1, PT;                      /* 0x190ec00004f3dc03 */
        /*1590*/                   IADD32I.X R15, R13, -0x1;                                   /* 0x0bfffffffcd3dc42 */
        /*1598*/                   SEL R19, R14, R19, !P1;                                     /* 0x201200004ce4dc04 */
        /*15a0*/                   SEL R18, R13, R18, !P1;                                     /* 0x2012000048d49c04 */
        /*15a8*/                   SEL R14, R16, R14, !P1;                                     /* 0x2012000039039c04 */
        /*15b0*/                   SEL R13, R15, R13, !P1;                                     /* 0x2012000034f35c04 */
        /*15b8*/                   MOV R16, R19;                                               /* 0x280000004c041de4 */
                                                                                               /* 0x220202d292621267 */
        /*15c8*/                   MOV R17, R18;                                               /* 0x2800000048045de4 */
        /*15d0*/                   MOV R15, R13;                                               /* 0x280000003403dde4 */
        /*15d8*/                   DFMA R16, R6, R16, -R8;                                     /* 0x2010000040641d01 */
        /*15e0*/                   DFMA R6, R6, R14, -R8;                                      /* 0x2010000038619d01 */
        /*15e8*/                   DSETP.GT.AND P1, PT, |R16|, |R6|, PT;                       /* 0x1a0e00001903dcc1 */
        /*15f0*/                   SEL R16, R14, R19, P1;                                      /* 0x200200004ce41c04 */
        /*15f8*/                   SEL R17, R13, R18, P1;                                      /* 0x2002000048d45c04 */
                                                                                               /* 0x22d2d28042b2d2e7 */
        /*1608*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1610*/                   DSETP.EQ.AND P1, PT, R20, RZ, PT;                           /* 0x190e0000fd43dc01 */
        /*1618*/               @P1 BRA 0x1678;                                                 /* 0x40000001600005e7 */
        /*1620*/                   MUFU.RCP64H R15, R7;                                        /* 0xc80000001873dc00 */
        /*1628*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
        /*1630*/                   DSETP.GT.AND P2, PT, |R14|, RZ, PT;                         /* 0x1a0e0000fce5dc81 */
        /*1638*/              @!P2 DSETP.NEU.AND P1, PT, |R6|, +INF , PT;                      /* 0x1e8edffc0063e881 */
                                                                                               /* 0x2202e20282027207 */
        /*1648*/              @!P2 SEL R6, R6, R14, P1;                                        /* 0x200200003861a804 */
        /*1650*/              @!P2 SEL R7, R7, R15, P1;                                        /* 0x200200003c71e804 */
        /*1658*/              @!P2 MOV R14, R6;                                                /* 0x280000001803a9e4 */
        /*1660*/              @!P2 MOV R15, R7;                                                /* 0x280000001c03e9e4 */
        /*1668*/                   DMUL R16, R8, R14;                                          /* 0x5000000038841c01 */
        /*1670*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1678*/                   DMUL R16, R8, R6;                                           /* 0x5000000018841c01 */
                                                                                               /* 0x22020042e20042b7 */
        /*1688*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1690*/                   MOV R6, R16;                                                /* 0x2800000040019de4 */
        /*1698*/                   MOV R7, R17;                                                /* 0x280000004401dde4 */
        /*16a0*/                   RET;                                                        /* 0x9000000000001de7 */
        /*16a8*/                   ISETP.LT.U32.AND P1, PT, R9, c[0x2][0x38], PT;              /* 0x188e4800e093dc03 */
        /*16b0*/                   MOV R9, R13;                                                /* 0x2800000034025de4 */
        /*16b8*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
                                                                                               /* 0x22d29282c292b2a7 */
        /*16c8*/                   PBK 0x1828;                                                 /* 0x6800000560001c07 */
        /*16d0*/               @P1 BRA 0x1720;                                                 /* 0x40000001200005e7 */
        /*16d8*/                   DFMA.RM R8, R14, R8, R16;                                   /* 0x20a0000020e21c01 */
        /*16e0*/                   IADD32I R18.CC, R8, 0x1;                                    /* 0x0c00000004849c02 */
        /*16e8*/                   IADD.X R19, RZ, R9;                                         /* 0x4800000027f4dc43 */
        /*16f0*/                   DFMA.RP R6, -R8, R18, R6;                                   /* 0x210c000048819e01 */
        /*16f8*/                   DSETP.GT.AND P1, PT, R6, RZ, PT;                            /* 0x1a0e0000fc63dc01 */
                                                                                               /* 0x2202c2b2d2e04207 */
        /*1708*/                   SEL R8, R18, R8, P1;                                        /* 0x2002000021221c04 */
        /*1710*/                   SEL R9, R19, R9, P1;                                        /* 0x2002000025325c04 */
        /*1718*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1720*/                   DSETP.EQU.AND P1, PT, R6, RZ, PT;                           /* 0x1d0e0000fc63dc01 */
        /*1728*/               @P1 BRA 0x1818;                                                 /* 0x40000003a00005e7 */
        /*1730*/                   ISETP.LT.AND P1, PT, R7, RZ, PT;                            /* 0x188e0000fc73dc23 */
        /*1738*/               @P1 MOV32I R8, 0x0;                                             /* 0x18000000000205e2 */
                                                                                               /* 0x22804202b2c2b047 */
        /*1748*/               @P1 MOV32I R9, 0xfff80000;                                      /* 0x1bffe000000245e2 */
        /*1750*/               @P1 BRK;                                                        /* 0xa8000000000005e7 */
        /*1758*/                   ISETP.GT.AND P1, PT, R7, c[0x2][0x3c], PT;                  /* 0x1a0e4800f073dc23 */
        /*1760*/               @P1 BRA 0x1818;                                                 /* 0x40000002c00005e7 */
        /*1768*/                   DMUL R6, R6, 8.11296384146066816958e+31;                    /* 0x5000d1a400619c01 */
        /*1770*/                   MOV R8, RZ;                                                 /* 0x28000000fc021de4 */
        /*1778*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
                                                                                               /* 0x2292923292928047 */
        /*1788*/                   MUFU.RSQ64H R9, R7;                                         /* 0xc80000001c725c00 */
        /*1790*/                   MOV32I R19, 0x3fe00000;                                     /* 0x18ff80000004dde2 */
        /*1798*/                   DMUL R16, R8, R8;                                           /* 0x5000000020841c01 */
        /*17a0*/                   DFMA R14, -R6, R16, c[0x2][0x0];                            /* 0x2020880000639e01 */
        /*17a8*/                   DFMA R18, R14, 0.375, R18;                                  /* 0x2024cff600e49c01 */
        /*17b0*/                   DMUL R14, R8, R14;                                          /* 0x5000000038839c01 */
        /*17b8*/                   DFMA R8, R18, R14, R8;                                      /* 0x2010000039221c01 */
                                                                                               /* 0x2202928042028207 */
        /*17c8*/                   DMUL R16, R6, R8;                                           /* 0x5000000020641c01 */
        /*17d0*/                   IADD32I R9, R9, -0x100000;                                  /* 0x0bffc00000925c02 */
        /*17d8*/                   DFMA R18, -R16, R16, R6;                                    /* 0x200c000041049e01 */
        /*17e0*/                   MOV R6, R8;                                                 /* 0x2800000020019de4 */
        /*17e8*/                   MOV R7, R9;                                                 /* 0x280000002401dde4 */
        /*17f0*/                   DFMA R6, R6, R18, R16;                                      /* 0x2020000048619c01 */
        /*17f8*/                   IADD32I R9, R7, -0x3500000;                                 /* 0x0bf2c00000725c02 */
                                                                                               /* 0x22e20042b202e047 */
        /*1808*/                   MOV R8, R6;                                                 /* 0x2800000018021de4 */
        /*1810*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1818*/                   DADD R8, R6, R6;                                            /* 0x4800000018621c01 */
        /*1820*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1828*/                   MOV R28, R8;                                                /* 0x2800000020071de4 */
        /*1830*/                   MOV R29, R9;                                                /* 0x2800000024075de4 */
        /*1838*/                   RET;                                                        /* 0x9000000000001de7 */
        /*1840*/                   BRA 0x1840;                                                 /* 0x4003ffffe0001de7 */
        /*1848*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*1850*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*1858*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*1860*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*1868*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*1870*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*1878*/                   NOP;                                                        /* 0x4000000000001de4 */
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
.reg .pred %p<7>;
.reg .b32 %r<21>;
.reg .f64 %fd<93>;
.reg .b64 %rd<52>;


mov.u32 %r1, %tid.x;
mov.u32 %r2, %ntid.x;
mov.u32 %r3, %ctaid.x;
mad.lo.s32 %r4, %r2, %r3, %r1;
cvt.s64.s32	%rd1, %r4;
ld.global.u64 %rd15, [__Bodydata_buffer];
setp.ge.u64	%p1, %rd1, %rd15;
@%p1 bra BB8_11;

add.s64 %rd2, %rd1, 1;
mov.u64 %rd51, 8000;
mov.u64 %rd50, __Bodydata_buffer;
mov.u64 %rd49, 4;
mov.u64 %rd48, 3;
mov.u64 %rd47, 2;
mov.u64 %rd46, 1;

BB8_2:
setp.eq.s64	%p2, %rd46, %rd2;
@%p2 bra BB8_4;

shl.b64 %rd22, %rd46, 3;
mov.u64 %rd23, __Bodydata_buffer;
add.s64 %rd24, %rd23, 16;
add.s64 %rd25, %rd22, %rd24;
mul.wide.s32 %rd26, %r4, 8;
add.s64 %rd27, %rd26, %rd24;
ld.global.f64 %fd1, [%rd27+64008];
ld.global.f64 %fd2, [%rd25+64000];
sub.f64 %fd3, %fd2, %fd1;
ld.global.f64 %fd4, [%rd27+128008];
ld.global.f64 %fd5, [%rd25+128000];
sub.f64 %fd6, %fd5, %fd4;
mul.f64 %fd7, %fd6, %fd6;
fma.rn.f64 %fd8, %fd3, %fd3, %fd7;
sqrt.rn.f64 %fd9, %fd8;
ld.global.f64 %fd10, [%rd27+8];
mul.f64 %fd11, %fd10, 0d3DD257B4D002790D;
ld.global.f64 %fd12, [%rd50+24];
mul.f64 %fd13, %fd11, %fd12;
fma.rn.f64 %fd14, %fd9, %fd9, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd15, %fd13, %fd14;
mul.f64 %fd16, %fd3, %fd15;
div.rn.f64 %fd17, %fd16, %fd9;
ld.global.f64 %fd18, [%rd27+320008];
add.f64 %fd19, %fd17, %fd18;
ld.global.f64 %fd20, [%rd27+384008];
st.global.f64 [%rd27+320008], %fd19;
mul.f64 %fd21, %fd6, %fd15;
div.rn.f64 %fd22, %fd21, %fd9;
add.f64 %fd23, %fd22, %fd20;
st.global.f64 [%rd27+384008], %fd23;

BB8_4:
setp.eq.s64	%p3, %rd47, %rd2;
@%p3 bra BB8_6;

shl.b64 %rd28, %rd47, 3;
mov.u64 %rd29, __Bodydata_buffer;
add.s64 %rd30, %rd29, 16;
add.s64 %rd31, %rd28, %rd30;
mul.wide.s32 %rd32, %r4, 8;
add.s64 %rd33, %rd32, %rd30;
ld.global.f64 %fd24, [%rd33+64008];
ld.global.f64 %fd25, [%rd31+64000];
sub.f64 %fd26, %fd25, %fd24;
ld.global.f64 %fd27, [%rd33+128008];
ld.global.f64 %fd28, [%rd31+128000];
sub.f64 %fd29, %fd28, %fd27;
mul.f64 %fd30, %fd29, %fd29;
fma.rn.f64 %fd31, %fd26, %fd26, %fd30;
sqrt.rn.f64 %fd32, %fd31;
ld.global.f64 %fd33, [%rd33+8];
mul.f64 %fd34, %fd33, 0d3DD257B4D002790D;
ld.global.f64 %fd35, [%rd50+32];
mul.f64 %fd36, %fd34, %fd35;
fma.rn.f64 %fd37, %fd32, %fd32, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd38, %fd36, %fd37;
mul.f64 %fd39, %fd26, %fd38;
div.rn.f64 %fd40, %fd39, %fd32;
ld.global.f64 %fd41, [%rd33+320008];
add.f64 %fd42, %fd40, %fd41;
ld.global.f64 %fd43, [%rd33+384008];
st.global.f64 [%rd33+320008], %fd42;
mul.f64 %fd44, %fd29, %fd38;
div.rn.f64 %fd45, %fd44, %fd32;
add.f64 %fd46, %fd45, %fd43;
st.global.f64 [%rd33+384008], %fd46;

BB8_6:
setp.eq.s64	%p4, %rd48, %rd2;
@%p4 bra BB8_8;

shl.b64 %rd34, %rd48, 3;
mov.u64 %rd35, __Bodydata_buffer;
add.s64 %rd36, %rd35, 16;
add.s64 %rd37, %rd34, %rd36;
mul.wide.s32 %rd38, %r4, 8;
add.s64 %rd39, %rd38, %rd36;
ld.global.f64 %fd47, [%rd39+64008];
ld.global.f64 %fd48, [%rd37+64000];
sub.f64 %fd49, %fd48, %fd47;
ld.global.f64 %fd50, [%rd39+128008];
ld.global.f64 %fd51, [%rd37+128000];
sub.f64 %fd52, %fd51, %fd50;
mul.f64 %fd53, %fd52, %fd52;
fma.rn.f64 %fd54, %fd49, %fd49, %fd53;
sqrt.rn.f64 %fd55, %fd54;
ld.global.f64 %fd56, [%rd39+8];
mul.f64 %fd57, %fd56, 0d3DD257B4D002790D;
ld.global.f64 %fd58, [%rd50+40];
mul.f64 %fd59, %fd57, %fd58;
fma.rn.f64 %fd60, %fd55, %fd55, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd61, %fd59, %fd60;
mul.f64 %fd62, %fd49, %fd61;
div.rn.f64 %fd63, %fd62, %fd55;
ld.global.f64 %fd64, [%rd39+320008];
add.f64 %fd65, %fd63, %fd64;
ld.global.f64 %fd66, [%rd39+384008];
st.global.f64 [%rd39+320008], %fd65;
mul.f64 %fd67, %fd52, %fd61;
div.rn.f64 %fd68, %fd67, %fd55;
add.f64 %fd69, %fd68, %fd66;
st.global.f64 [%rd39+384008], %fd69;

BB8_8:
setp.eq.s64	%p5, %rd49, %rd2;
@%p5 bra BB8_10;

shl.b64 %rd40, %rd49, 3;
mov.u64 %rd41, __Bodydata_buffer;
add.s64 %rd42, %rd41, 16;
add.s64 %rd43, %rd40, %rd42;
mul.wide.s32 %rd44, %r4, 8;
add.s64 %rd45, %rd44, %rd42;
ld.global.f64 %fd70, [%rd45+64008];
ld.global.f64 %fd71, [%rd43+64000];
sub.f64 %fd72, %fd71, %fd70;
ld.global.f64 %fd73, [%rd45+128008];
ld.global.f64 %fd74, [%rd43+128000];
sub.f64 %fd75, %fd74, %fd73;
mul.f64 %fd76, %fd75, %fd75;
fma.rn.f64 %fd77, %fd72, %fd72, %fd76;
sqrt.rn.f64 %fd78, %fd77;
ld.global.f64 %fd79, [%rd45+8];
mul.f64 %fd80, %fd79, 0d3DD257B4D002790D;
ld.global.f64 %fd81, [%rd50+48];
mul.f64 %fd82, %fd80, %fd81;
fma.rn.f64 %fd83, %fd78, %fd78, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd84, %fd82, %fd83;
mul.f64 %fd85, %fd72, %fd84;
div.rn.f64 %fd86, %fd85, %fd78;
ld.global.f64 %fd87, [%rd45+320008];
add.f64 %fd88, %fd86, %fd87;
ld.global.f64 %fd89, [%rd45+384008];
st.global.f64 [%rd45+320008], %fd88;
mul.f64 %fd90, %fd75, %fd84;
div.rn.f64 %fd91, %fd90, %fd78;
add.f64 %fd92, %fd91, %fd89;
st.global.f64 [%rd45+384008], %fd92;

BB8_10:
add.s64 %rd51, %rd51, -4;
add.s64 %rd50, %rd50, 32;
add.s64 %rd49, %rd49, 4;
add.s64 %rd48, %rd48, 4;
add.s64 %rd47, %rd47, 4;
add.s64 %rd46, %rd46, 4;
setp.ne.s64	%p6, %rd51, 0;
@%p6 bra BB8_2;

BB8_11:
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
add.s64 %rd7, %rd5, %rd6;
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
.reg .b64 %rd<10>;


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
add.s64 %rd9, %rd7, %rd8;
add.s64 %rd3, %rd9, 24;
add.s64 %rd2, %rd9, 320024;
add.s64 %rd4, %rd9, 192024;
add.s64 %rd5, %rd9, 64024;
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


