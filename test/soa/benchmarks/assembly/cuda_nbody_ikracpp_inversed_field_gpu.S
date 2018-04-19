
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
  GLOBAL:448496
 Function _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I4BodyLm0EEEJEEEvPT_DpT0_:
  REG:4 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I4BodyLm0EEEEEvPT_m:
  REG:10 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I4BodyLm0EEEEEvPT_m:
  REG:10 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_22codengen_simple_updateEdEEE4callEdEUldE_JdEEEvT_DpT0_:
  REG:22 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:32 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvdEXadL_ZNS4_6updateEdEEE4callEdEUldE_JdEEEvT_DpT0_:
  REG:32 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:32 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_22add_all_forces_to_thisEvEEE4callEvEUlvE_JEEEvT_DpT0_:
  REG:54 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:321 CONSTANT[2]:60 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM4BodyFvvEXadL_ZNS4_11reset_forceEvEEE4callEvEUlvE_JEEEvT_DpT0_:
  REG:8 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:321 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z22codengen_simple_updateP4Bodyd:
  REG:22 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:32 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z18codegen_reset_massP4Body:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z24codegen_reset_force_halfP4Body:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z26codegen_reset_force_manualP4Body:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z14codegen_updateP4Bodyd:
  REG:32 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:32 TEXTURE:0 SURFACE:0 SAMPLER:0
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
        /*0048*/                   ST.E [R2+0x8], RZ;                             /* 0x94000000202fdc85 */
        /*0050*/                   EXIT;                                          /* 0x8000000000001de7 */
        /*0058*/                   BRA 0x58;                                      /* 0x4003ffffe0001de7 */
        /*0060*/                   NOP;                                           /* 0x4000000000001de4 */
        /*0068*/                   NOP;                                           /* 0x4000000000001de4 */
        /*0070*/                   NOP;                                           /* 0x4000000000001de4 */
        /*0078*/                   NOP;                                           /* 0x4000000000001de4 */
		.....................................................................................................


		Function : _ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I4BodyLm0EEEEEvPT_m
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
		.............................................................................................


		Function : _ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I4BodyLm0EEEEEvPT_m
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                         /* 0x22b2004282004307 */
        /*0008*/                   MOV R1, c[0x0][0x44];                 /* 0x2800400110005de4 */
        /*0010*/                   MOV R6, c[0x0][0x140];                /* 0x2800400500019de4 */
        /*0018*/                   S2R R3, SR_LANEID;                    /* 0x2c0000000000dc04 */
        /*0020*/                   VOTE.ALL R0, PT, PT;                  /* 0x49c0000000701c04 */
        /*0028*/                   IADD32I R6.CC, R6, 0x8;               /* 0x0c00000020619c02 */
        /*0030*/                   POPC R2, R0, R0;                      /* 0x5400000000009c04 */
        /*0038*/                   FLO.U32 R5, R0;                       /* 0x7800000000015c03 */
                                                                         /* 0x2202804240423047 */
        /*0048*/                   IADD.X R7, RZ, c[0x0][0x144];         /* 0x4800400513f1dc43 */
        /*0050*/                   ISETP.EQ.U32.AND P0, PT, R5, R3, PT;  /* 0x190e00000c51dc03 */
        /*0058*/                   IMUL R4, R2, c[0x0][0x148];           /* 0x5000400520211ca3 */
        /*0060*/                   MOV R2, R6;                           /* 0x2800000018009de4 */
        /*0068*/                   MOV R3, R7;                           /* 0x280000001c00dde4 */
        /*0070*/                   S2R R6, SR_LTMASK;                    /* 0x2c000000e4019c04 */
        /*0078*/               @P0 ATOM.E.ADD R4, [R2], R4;              /* 0x547e200000210005 */
                                                                         /* 0x22c2804373f042d7 */
        /*0088*/                   LOP.AND R0, R6, R0;                   /* 0x6800000000601c03 */
        /*0090*/                   POPC R0, R0, R0;                      /* 0x5400000000001c04 */
        /*0098*/                   CCTL.E.IV [R2];                       /* 0x9c00000000201ca5 */
        /*00a0*/                   SHFL.IDX PT, R5, R4, R5, 0x1f;        /* 0x8c007c0014415f45 */
        /*00a8*/                   IMAD R0, R0, c[0x0][0x148], R5;       /* 0x200a400520001ca3 */
        /*00b0*/                   MOV32I R5, 0x0;                       /* 0x1800000000015de2 */
        /*00b8*/                   IADD R0.CC, R0, c[0x0][0x140];        /* 0x4801400500001c03 */
                                                                         /* 0x22f0428042b20047 */
        /*00c8*/                   IADD.X R9, RZ, c[0x0][0x144];         /* 0x4800400513f25c43 */
        /*00d0*/                   IADD32I R8.CC, R0, 0x8;               /* 0x0c00000020021c02 */
        /*00d8*/                   MOV32I R0, 0x0;                       /* 0x1800000000001de2 */
        /*00e0*/                   MOV R4, R0;                           /* 0x2800000000011de4 */
        /*00e8*/                   IADD.X R9, RZ, R9;                    /* 0x4800000027f25c43 */
        /*00f0*/                   ST.E.64 [R4], R8;                     /* 0x9400000000421ca5 */
        /*00f8*/                   EXIT;                                 /* 0x8000000000001de7 */
        /*0100*/                   BRA 0x100;                            /* 0x4003ffffe0001de7 */
        /*0108*/                   NOP;                                  /* 0x4000000000001de4 */
        /*0110*/                   NOP;                                  /* 0x4000000000001de4 */
        /*0118*/                   NOP;                                  /* 0x4000000000001de4 */
        /*0120*/                   NOP;                                  /* 0x4000000000001de4 */
        /*0128*/                   NOP;                                  /* 0x4000000000001de4 */
        /*0130*/                   NOP;                                  /* 0x4000000000001de4 */
        /*0138*/                   NOP;                                  /* 0x4000000000001de4 */
		.........................................................................................


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
        /*00e0*/                   LD.E.64 R8, [R2+0x18];                                     /* 0x8400000060221ca5 */
        /*00e8*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*00f0*/                   LD.E.64 R4, [R2+0x4e358];                                  /* 0x8400138d60211ca5 */
        /*00f8*/                   SSY 0x1b8;                                                 /* 0x60000002e0001c07 */
                                                                                              /* 0x2292929252329287 */
        /*0108*/                   MUFU.RCP64H R11, R9;                                       /* 0xc80000001892dc00 */
        /*0110*/                   DFMA R6, -R8, R10, c[0x2][0x0];                            /* 0x2014880000819e01 */
        /*0118*/                   DFMA R12, R6, R6, R6;                                      /* 0x200c000018631c01 */
        /*0120*/                   DMUL R6, R4, c[0x0][0x148];                                /* 0x5000400520419c01 */
        /*0128*/                   DFMA R4, R10, R12, R10;                                    /* 0x2014000030a11c01 */
        /*0130*/                   DMUL R10, R6, R4;                                          /* 0x5000000010629c01 */
        /*0138*/                   DFMA R14, -R8, R10, R6;                                    /* 0x200c000028839e01 */
                                                                                              /* 0x220042b2c2828207 */
        /*0148*/                   DFMA R4, R4, R14, R10;                                     /* 0x2014000038411c01 */
        /*0150*/                   FSET.GTU.AND RZ.CC, |R7|, c[0x2][0x8], PT;                 /* 0x160f4800207fdc80 */
        /*0158*/                   FFMA R10, RZ, R9, R5;                                      /* 0x300a000027f29c00 */
        /*0160*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*0168*/               @P0 BRA CC.NEU, 0x1b0;                                         /* 0x40000001000001a7 */
        /*0170*/                   MOV R4, R6;                                                /* 0x2800000018011de4 */
        /*0178*/                   MOV R5, R7;                                                /* 0x280000001c015de4 */
                                                                                              /* 0x2232b2b042b2b047 */
        /*0188*/                   MOV R6, R8;                                                /* 0x2800000020019de4 */
        /*0190*/                   MOV R7, R9;                                                /* 0x280000002401dde4 */
        /*0198*/                   CAL 0xb28;                                                 /* 0x5000002620011c07 */
        /*01a0*/                   MOV R4, R8;                                                /* 0x2800000020011de4 */
        /*01a8*/                   MOV R5, R9;                                                /* 0x2800000024015de4 */
        /*01b0*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*01b8*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
                                                                                              /* 0x22329283f0423047 */
        /*01c8*/                   LD.E.64 R10, [R2+0x5dd98];                                 /* 0x8400177660229ca5 */
        /*01d0*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*01d8*/                   LD.E.64 R8, [R2+0x2eed8];                                  /* 0x84000bbb60221ca5 */
        /*01e0*/                   SSY 0x2a0;                                                 /* 0x60000002e0001c07 */
        /*01e8*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*01f0*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*01f8*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
                                                                                              /* 0x2232523292929257 */
        /*0208*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*0210*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*0218*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0220*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
        /*0228*/                   DFMA R18, R12, R14, R16;                                   /* 0x2020000038c49c01 */
        /*0230*/                   DADD R4, R4, R8;                                           /* 0x4800000020411c01 */
        /*0238*/                   FFMA R8, RZ, R7, R19;                                      /* 0x302600001ff21c00 */
                                                                                              /* 0x22b2b042b2c24047 */
        /*0248*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*0250*/                   ST.E.64 [R2+0x2eed8], R4;                                  /* 0x94000bbb60211ca5 */
        /*0258*/                   FSETP.GT.AND P0, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003081dc80 */
        /*0260*/               @P0 BRA CC.NEU, 0x298;                                         /* 0x40000000c00001a7 */
        /*0268*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*0270*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*0278*/                   CAL 0xb28;                                                 /* 0x50000022a0011c07 */
                                                                                              /* 0x2232323042b2b047 */
        /*0288*/                   MOV R18, R8;                                               /* 0x2800000020049de4 */
        /*0290*/                   MOV R19, R9;                                               /* 0x280000002404dde4 */
        /*0298*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*02a0*/                   LD.E.64 R12, [R2+0x3e918];                                 /* 0x84000fa460231ca5 */
        /*02a8*/                   SSY 0x3b8;                                                 /* 0x6000000420001c07 */
        /*02b0*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
        /*02b8*/                   LD.E.64 R16, [R2+0x4e358];                                 /* 0x8400138d60241ca5 */
                                                                                              /* 0x22e042e04203f237 */
        /*02c8*/                   LD.E.64 R4, [R2+0x2eed8];                                  /* 0x84000bbb60211ca5 */
        /*02d0*/                   LD.E.64 R8, [R2+0xfa58];                                   /* 0x840003e960221ca5 */
        /*02d8*/                   DADD R14, R18, R12;                                        /* 0x4800000031239c01 */
        /*02e0*/                   LD.E.64 R12, [R2+0x1f498];                                 /* 0x840007d260231ca5 */
        /*02e8*/                   MOV32I R18, 0x1;                                           /* 0x1800000004049de2 */
        /*02f0*/                   ST.E.64 [R2+0x3e918], R14;                                 /* 0x94000fa460239ca5 */
        /*02f8*/                   MUFU.RCP64H R19, R7;                                       /* 0xc80000001874dc00 */
                                                                                              /* 0x2252329252329047 */
        /*0308*/                   LD.E.64 R10, [R2+0x3e918];                                 /* 0x84000fa460229ca5 */
        /*0310*/                   DFMA R20, -R6, R18, c[0x2][0x0];                           /* 0x2024880000651e01 */
        /*0318*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*0320*/                   DMUL R16, R16, c[0x0][0x148];                              /* 0x5000400521041c01 */
        /*0328*/                   DFMA R18, R18, R20, R18;                                   /* 0x2024000051249c01 */
        /*0330*/                   DMUL R20, R16, R18;                                        /* 0x5000000049051c01 */
        /*0338*/                   DFMA R4, R4, c[0x0][0x148], R8;                            /* 0x2010400520411c01 */
                                                                                              /* 0x22028203f0420297 */
        /*0348*/                   DFMA R8, -R6, R20, R16;                                    /* 0x2020000050621e01 */
        /*0350*/                   DFMA R8, R18, R8, R20;                                     /* 0x2028000021221c01 */
        /*0358*/                   ST.E.64 [R2+0xfa58], R4;                                   /* 0x940003e960211ca5 */
        /*0360*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                /* 0x160f4800211fdc80 */
        /*0368*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*0370*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*0378*/                   ST.E.64 [R2+0x1f498], R12;                                 /* 0x940007d260231ca5 */
                                                                                              /* 0x2232b2b2b042b2c7 */
        /*0388*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*0390*/               @P0 BRA CC.NEU, 0x3b0;                                         /* 0x40000000600001a7 */
        /*0398*/                   MOV R4, R16;                                               /* 0x2800000040011de4 */
        /*03a0*/                   MOV R5, R17;                                               /* 0x2800000044015de4 */
        /*03a8*/                   CAL 0xb28;                                                 /* 0x5000001de0011c07 */
        /*03b0*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*03b8*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
                                                                                              /* 0x22329283f0423047 */
        /*03c8*/                   LD.E.64 R10, [R2+0x5dd98];                                 /* 0x8400177660229ca5 */
        /*03d0*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*03d8*/                   LD.E.64 R4, [R2+0x2eed8];                                  /* 0x84000bbb60211ca5 */
        /*03e0*/                   SSY 0x4a0;                                                 /* 0x60000002e0001c07 */
        /*03e8*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*03f0*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*03f8*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
                                                                                              /* 0x2232523292929257 */
        /*0408*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*0410*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*0418*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0420*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
        /*0428*/                   DFMA R18, R12, R14, R16;                                   /* 0x2020000038c49c01 */
        /*0430*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
        /*0438*/                   FFMA R8, RZ, R7, R19;                                      /* 0x302600001ff21c00 */
                                                                                              /* 0x22b2b042b2c24047 */
        /*0448*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*0450*/                   ST.E.64 [R2+0x2eed8], R4;                                  /* 0x94000bbb60211ca5 */
        /*0458*/                   FSETP.GT.AND P0, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003081dc80 */
        /*0460*/               @P0 BRA CC.NEU, 0x498;                                         /* 0x40000000c00001a7 */
        /*0468*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*0470*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*0478*/                   CAL 0xb28;                                                 /* 0x5000001aa0011c07 */
                                                                                              /* 0x2232323042b2b047 */
        /*0488*/                   MOV R18, R8;                                               /* 0x2800000020049de4 */
        /*0490*/                   MOV R19, R9;                                               /* 0x280000002404dde4 */
        /*0498*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*04a0*/                   LD.E.64 R12, [R2+0x3e918];                                 /* 0x84000fa460231ca5 */
        /*04a8*/                   SSY 0x5b8;                                                 /* 0x6000000420001c07 */
        /*04b0*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
        /*04b8*/                   LD.E.64 R16, [R2+0x4e358];                                 /* 0x8400138d60241ca5 */
                                                                                              /* 0x22e042e04203f237 */
        /*04c8*/                   LD.E.64 R4, [R2+0x2eed8];                                  /* 0x84000bbb60211ca5 */
        /*04d0*/                   LD.E.64 R8, [R2+0xfa58];                                   /* 0x840003e960221ca5 */
        /*04d8*/                   DADD R14, R18, R12;                                        /* 0x4800000031239c01 */
        /*04e0*/                   LD.E.64 R12, [R2+0x1f498];                                 /* 0x840007d260231ca5 */
        /*04e8*/                   MOV32I R18, 0x1;                                           /* 0x1800000004049de2 */
        /*04f0*/                   ST.E.64 [R2+0x3e918], R14;                                 /* 0x94000fa460239ca5 */
        /*04f8*/                   MUFU.RCP64H R19, R7;                                       /* 0xc80000001874dc00 */
                                                                                              /* 0x2252329252329047 */
        /*0508*/                   LD.E.64 R10, [R2+0x3e918];                                 /* 0x84000fa460229ca5 */
        /*0510*/                   DFMA R20, -R6, R18, c[0x2][0x0];                           /* 0x2024880000651e01 */
        /*0518*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*0520*/                   DMUL R16, R16, c[0x0][0x148];                              /* 0x5000400521041c01 */
        /*0528*/                   DFMA R18, R18, R20, R18;                                   /* 0x2024000051249c01 */
        /*0530*/                   DMUL R20, R16, R18;                                        /* 0x5000000049051c01 */
        /*0538*/                   DFMA R4, R4, c[0x0][0x148], R8;                            /* 0x2010400520411c01 */
                                                                                              /* 0x22028203f0420297 */
        /*0548*/                   DFMA R8, -R6, R20, R16;                                    /* 0x2020000050621e01 */
        /*0550*/                   DFMA R8, R18, R8, R20;                                     /* 0x2028000021221c01 */
        /*0558*/                   ST.E.64 [R2+0xfa58], R4;                                   /* 0x940003e960211ca5 */
        /*0560*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                /* 0x160f4800211fdc80 */
        /*0568*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*0570*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*0578*/                   ST.E.64 [R2+0x1f498], R12;                                 /* 0x940007d260231ca5 */
                                                                                              /* 0x2232b2b2b042b2c7 */
        /*0588*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*0590*/               @P0 BRA CC.NEU, 0x5b0;                                         /* 0x40000000600001a7 */
        /*0598*/                   MOV R4, R16;                                               /* 0x2800000040011de4 */
        /*05a0*/                   MOV R5, R17;                                               /* 0x2800000044015de4 */
        /*05a8*/                   CAL 0xb28;                                                 /* 0x50000015e0011c07 */
        /*05b0*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*05b8*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
                                                                                              /* 0x22329283f0423047 */
        /*05c8*/                   LD.E.64 R10, [R2+0x5dd98];                                 /* 0x8400177660229ca5 */
        /*05d0*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*05d8*/                   LD.E.64 R4, [R2+0x2eed8];                                  /* 0x84000bbb60211ca5 */
        /*05e0*/                   SSY 0x6a0;                                                 /* 0x60000002e0001c07 */
        /*05e8*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*05f0*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*05f8*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
                                                                                              /* 0x2232523292929257 */
        /*0608*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*0610*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*0618*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0620*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
        /*0628*/                   DFMA R18, R12, R14, R16;                                   /* 0x2020000038c49c01 */
        /*0630*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
        /*0638*/                   FFMA R8, RZ, R7, R19;                                      /* 0x302600001ff21c00 */
                                                                                              /* 0x22b2b042b2c24047 */
        /*0648*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*0650*/                   ST.E.64 [R2+0x2eed8], R4;                                  /* 0x94000bbb60211ca5 */
        /*0658*/                   FSETP.GT.AND P0, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003081dc80 */
        /*0660*/               @P0 BRA CC.NEU, 0x698;                                         /* 0x40000000c00001a7 */
        /*0668*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*0670*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*0678*/                   CAL 0xb28;                                                 /* 0x50000012a0011c07 */
                                                                                              /* 0x2232323042b2b047 */
        /*0688*/                   MOV R18, R8;                                               /* 0x2800000020049de4 */
        /*0690*/                   MOV R19, R9;                                               /* 0x280000002404dde4 */
        /*0698*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*06a0*/                   LD.E.64 R12, [R2+0x3e918];                                 /* 0x84000fa460231ca5 */
        /*06a8*/                   SSY 0x7b8;                                                 /* 0x6000000420001c07 */
        /*06b0*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
        /*06b8*/                   LD.E.64 R16, [R2+0x4e358];                                 /* 0x8400138d60241ca5 */
                                                                                              /* 0x22e042e04203f237 */
        /*06c8*/                   LD.E.64 R4, [R2+0x2eed8];                                  /* 0x84000bbb60211ca5 */
        /*06d0*/                   LD.E.64 R8, [R2+0xfa58];                                   /* 0x840003e960221ca5 */
        /*06d8*/                   DADD R14, R18, R12;                                        /* 0x4800000031239c01 */
        /*06e0*/                   LD.E.64 R12, [R2+0x1f498];                                 /* 0x840007d260231ca5 */
        /*06e8*/                   MOV32I R18, 0x1;                                           /* 0x1800000004049de2 */
        /*06f0*/                   ST.E.64 [R2+0x3e918], R14;                                 /* 0x94000fa460239ca5 */
        /*06f8*/                   MUFU.RCP64H R19, R7;                                       /* 0xc80000001874dc00 */
                                                                                              /* 0x2252329252329047 */
        /*0708*/                   LD.E.64 R10, [R2+0x3e918];                                 /* 0x84000fa460229ca5 */
        /*0710*/                   DFMA R20, -R6, R18, c[0x2][0x0];                           /* 0x2024880000651e01 */
        /*0718*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*0720*/                   DMUL R16, R16, c[0x0][0x148];                              /* 0x5000400521041c01 */
        /*0728*/                   DFMA R18, R18, R20, R18;                                   /* 0x2024000051249c01 */
        /*0730*/                   DMUL R20, R16, R18;                                        /* 0x5000000049051c01 */
        /*0738*/                   DFMA R4, R4, c[0x0][0x148], R8;                            /* 0x2010400520411c01 */
                                                                                              /* 0x22028203f0420297 */
        /*0748*/                   DFMA R8, -R6, R20, R16;                                    /* 0x2020000050621e01 */
        /*0750*/                   DFMA R8, R18, R8, R20;                                     /* 0x2028000021221c01 */
        /*0758*/                   ST.E.64 [R2+0xfa58], R4;                                   /* 0x940003e960211ca5 */
        /*0760*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                /* 0x160f4800211fdc80 */
        /*0768*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*0770*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*0778*/                   ST.E.64 [R2+0x1f498], R12;                                 /* 0x940007d260231ca5 */
                                                                                              /* 0x2232b2b2b042b2c7 */
        /*0788*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*0790*/               @P0 BRA CC.NEU, 0x7b0;                                         /* 0x40000000600001a7 */
        /*0798*/                   MOV R4, R16;                                               /* 0x2800000040011de4 */
        /*07a0*/                   MOV R5, R17;                                               /* 0x2800000044015de4 */
        /*07a8*/                   CAL 0xb28;                                                 /* 0x5000000de0011c07 */
        /*07b0*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*07b8*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
                                                                                              /* 0x22329283f0423047 */
        /*07c8*/                   LD.E.64 R10, [R2+0x5dd98];                                 /* 0x8400177660229ca5 */
        /*07d0*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*07d8*/                   LD.E.64 R4, [R2+0x2eed8];                                  /* 0x84000bbb60211ca5 */
        /*07e0*/                   SSY 0x8a0;                                                 /* 0x60000002e0001c07 */
        /*07e8*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*07f0*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*07f8*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
                                                                                              /* 0x2232523292929257 */
        /*0808*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*0810*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*0818*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0820*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
        /*0828*/                   DFMA R18, R12, R14, R16;                                   /* 0x2020000038c49c01 */
        /*0830*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
        /*0838*/                   FFMA R8, RZ, R7, R19;                                      /* 0x302600001ff21c00 */
                                                                                              /* 0x22b2b042b2c24047 */
        /*0848*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*0850*/                   ST.E.64 [R2+0x2eed8], R4;                                  /* 0x94000bbb60211ca5 */
        /*0858*/                   FSETP.GT.AND P0, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003081dc80 */
        /*0860*/               @P0 BRA CC.NEU, 0x898;                                         /* 0x40000000c00001a7 */
        /*0868*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*0870*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*0878*/                   CAL 0xb28;                                                 /* 0x5000000aa0011c07 */
                                                                                              /* 0x2232323042b2b047 */
        /*0888*/                   MOV R18, R8;                                               /* 0x2800000020049de4 */
        /*0890*/                   MOV R19, R9;                                               /* 0x280000002404dde4 */
        /*0898*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*08a0*/                   LD.E.64 R12, [R2+0x3e918];                                 /* 0x84000fa460231ca5 */
        /*08a8*/                   SSY 0x9b8;                                                 /* 0x6000000420001c07 */
        /*08b0*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
        /*08b8*/                   LD.E.64 R16, [R2+0x4e358];                                 /* 0x8400138d60241ca5 */
                                                                                              /* 0x22e042e04203f237 */
        /*08c8*/                   LD.E.64 R4, [R2+0x2eed8];                                  /* 0x84000bbb60211ca5 */
        /*08d0*/                   LD.E.64 R8, [R2+0xfa58];                                   /* 0x840003e960221ca5 */
        /*08d8*/                   DADD R14, R18, R12;                                        /* 0x4800000031239c01 */
        /*08e0*/                   LD.E.64 R12, [R2+0x1f498];                                 /* 0x840007d260231ca5 */
        /*08e8*/                   MOV32I R18, 0x1;                                           /* 0x1800000004049de2 */
        /*08f0*/                   ST.E.64 [R2+0x3e918], R14;                                 /* 0x94000fa460239ca5 */
        /*08f8*/                   MUFU.RCP64H R19, R7;                                       /* 0xc80000001874dc00 */
                                                                                              /* 0x2252329252329047 */
        /*0908*/                   LD.E.64 R10, [R2+0x3e918];                                 /* 0x84000fa460229ca5 */
        /*0910*/                   DFMA R20, -R6, R18, c[0x2][0x0];                           /* 0x2024880000651e01 */
        /*0918*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*0920*/                   DMUL R16, R16, c[0x0][0x148];                              /* 0x5000400521041c01 */
        /*0928*/                   DFMA R18, R18, R20, R18;                                   /* 0x2024000051249c01 */
        /*0930*/                   DMUL R20, R16, R18;                                        /* 0x5000000049051c01 */
        /*0938*/                   DFMA R4, R4, c[0x0][0x148], R8;                            /* 0x2010400520411c01 */
                                                                                              /* 0x22028203f0420297 */
        /*0948*/                   DFMA R8, -R6, R20, R16;                                    /* 0x2020000050621e01 */
        /*0950*/                   DFMA R8, R18, R8, R20;                                     /* 0x2028000021221c01 */
        /*0958*/                   ST.E.64 [R2+0xfa58], R4;                                   /* 0x940003e960211ca5 */
        /*0960*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                /* 0x160f4800211fdc80 */
        /*0968*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*0970*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*0978*/                   ST.E.64 [R2+0x1f498], R12;                                 /* 0x940007d260231ca5 */
                                                                                              /* 0x2232b2b2b042b2c7 */
        /*0988*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*0990*/               @P0 BRA CC.NEU, 0x9b0;                                         /* 0x40000000600001a7 */
        /*0998*/                   MOV R4, R16;                                               /* 0x2800000040011de4 */
        /*09a0*/                   MOV R5, R17;                                               /* 0x2800000044015de4 */
        /*09a8*/                   CAL 0xb28;                                                 /* 0x50000005e0011c07 */
        /*09b0*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*09b8*/                   LD.E.64 R6, [R2+0x18];                                     /* 0x8400000060219ca5 */
                                                                                              /* 0x22329283f0423047 */
        /*09c8*/                   LD.E.64 R10, [R2+0x5dd98];                                 /* 0x8400177660229ca5 */
        /*09d0*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*09d8*/                   LD.E.64 R4, [R2+0x2eed8];                                  /* 0x84000bbb60211ca5 */
        /*09e0*/                   SSY 0xaa0;                                                 /* 0x60000002e0001c07 */
        /*09e8*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*09f0*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*09f8*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
                                                                                              /* 0x2232523292929257 */
        /*0a08*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*0a10*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*0a18*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0a20*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
        /*0a28*/                   DFMA R16, R12, R14, R16;                                   /* 0x2020000038c41c01 */
        /*0a30*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
        /*0a38*/                   FFMA R8, RZ, R7, R17;                                      /* 0x302200001ff21c00 */
                                                                                              /* 0x22b2b042b2c24047 */
        /*0a48*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*0a50*/                   ST.E.64 [R2+0x2eed8], R4;                                  /* 0x94000bbb60211ca5 */
        /*0a58*/                   FSETP.GT.AND P0, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003081dc80 */
        /*0a60*/               @P0 BRA CC.NEU, 0xa98;                                         /* 0x40000000c00001a7 */
        /*0a68*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*0a70*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*0a78*/                   CAL 0xb28;                                                 /* 0x50000002a0011c07 */
                                                                                              /* 0x2200428042b2b047 */
        /*0a88*/                   MOV R16, R8;                                               /* 0x2800000020041de4 */
        /*0a90*/                   MOV R17, R9;                                               /* 0x2800000024045de4 */
        /*0a98*/                   NOP.S;                                                     /* 0x4000000000001df4 */
        /*0aa0*/                   LD.E.64 R4, [R2+0x3e918];                                  /* 0x84000fa460211ca5 */
        /*0aa8*/                   IADD32I R0, R0, 0x5;                                       /* 0x0800000014001c02 */
        /*0ab0*/                   LD.E.64 R6, [R2+0x2eed8];                                  /* 0x84000bbb60219ca5 */
        /*0ab8*/                   ISETP.NE.AND P0, PT, R0, RZ, PT;                           /* 0x1a8e0000fc01dc23 */
                                                                                              /* 0x23f2e2e04293f237 */
        /*0ac8*/                   LD.E.64 R8, [R2+0xfa58];                                   /* 0x840003e960221ca5 */
        /*0ad0*/                   LD.E.64 R12, [R2+0x1f498];                                 /* 0x840007d260231ca5 */
        /*0ad8*/                   DADD R4, R16, R4;                                          /* 0x4800000011011c01 */
        /*0ae0*/                   ST.E.64 [R2+0x3e918], R4;                                  /* 0x94000fa460211ca5 */
        /*0ae8*/                   DFMA R6, R6, c[0x0][0x148], R8;                            /* 0x2010400520619c01 */
        /*0af0*/                   LD.E.64 R10, [R2+0x3e918];                                 /* 0x84000fa460229ca5 */
        /*0af8*/                   ST.E.64 [R2+0xfa58], R6;                                   /* 0x940003e960219ca5 */
                                                                                              /* 0x22028042e2e04297 */
        /*0b08*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*0b10*/                   ST.E.64 [R2+0x1f498], R12;                                 /* 0x940007d260231ca5 */
        /*0b18*/               @P0 BRA 0xe0;                                                  /* 0x4003ffd7000001e7 */
        /*0b20*/                   EXIT;                                                      /* 0x8000000000001de7 */
        /*0b28*/                   LOP32I.AND R8, R7, 0x40000000;                             /* 0x3900000000721c02 */
        /*0b30*/                   MOV32I R12, 0x5ff00000;                                    /* 0x197fc00000031de2 */
        /*0b38*/                   ISETP.LT.U32.AND P0, PT, R8, c[0x2][0x10], PT;             /* 0x188e48004081dc03 */
                                                                                              /* 0x228042028042b047 */
        /*0b48*/                   MOV R8, RZ;                                                /* 0x28000000fc021de4 */
        /*0b50*/                   LOP32I.AND R13, R5, 0x7f800000;                            /* 0x39fe000000535c02 */
        /*0b58*/                   SEL R9, R12, c[0x2][0x14], P0;                             /* 0x2000480050c25c04 */
        /*0b60*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0b68*/                   DMUL R10, R6, R8;                                          /* 0x5000000020629c01 */
        /*0b70*/                   ISETP.LT.U32.AND P0, PT, R13, c[0x2][0x18], PT;            /* 0x188e480060d1dc03 */
        /*0b78*/                   MOV R16, RZ;                                               /* 0x28000000fc041de4 */
                                                                                              /* 0x2292523282028047 */
        /*0b88*/                   MUFU.RCP64H R15, R11;                                      /* 0xc800000018b3dc00 */
        /*0b90*/                   PBK 0xe20;                                                 /* 0x6800000a20001c07 */
        /*0b98*/                   DFMA R18, -R10, R14, c[0x2][0x0];                          /* 0x201c880000a49e01 */
        /*0ba0*/                   SEL R17, R12, c[0x2][0x14], P0;                            /* 0x2000480050c45c04 */
        /*0ba8*/                   DFMA R18, R18, R18, R18;                                   /* 0x2024000049249c01 */
        /*0bb0*/                   DMUL R16, R4, R16;                                         /* 0x5000000040441c01 */
        /*0bb8*/                   DFMA R14, R14, R18, R14;                                   /* 0x201c000048e39c01 */
                                                                                              /* 0x22c042b2d2929297 */
        /*0bc8*/                   DMUL R20, R16, R14;                                        /* 0x5000000039051c01 */
        /*0bd0*/                   DFMA R10, -R10, R20, R16;                                  /* 0x2020000050a29e01 */
        /*0bd8*/                   DFMA R14, R14, R10, R20;                                   /* 0x2028000028e39c01 */
        /*0be0*/                   DSETP.LEU.AND P0, PT, |R14|, RZ, PT;                       /* 0x1d8e0000fce1dc81 */
        /*0be8*/               @P0 BRA 0xda0;                                                 /* 0x40000006c00001e7 */
        /*0bf0*/                   ISETP.GT.U32.AND P0, PT, R13, c[0x2][0x1c], PT;            /* 0x1a0e480070d1dc03 */
        /*0bf8*/                   MOV R10, RZ;                                               /* 0x28000000fc029de4 */
                                                                                              /* 0x2292523252327207 */
        /*0c08*/                   SEL R11, R12, c[0x2][0x14], P0;                            /* 0x2000480050c2dc04 */
        /*0c10*/                   DMUL R12, R8, R14;                                         /* 0x5000000038831c01 */
        /*0c18*/                   DMUL R14, R14, R10;                                        /* 0x5000000028e39c01 */
        /*0c20*/                   DMUL R10, R10, R12;                                        /* 0x5000000030a29c01 */
        /*0c28*/                   DMUL R14, R8, R14;                                         /* 0x5000000038839c01 */
        /*0c30*/                   DFMA R8, R6, R10, -R4;                                     /* 0x2008000028621d01 */
        /*0c38*/                   DFMA R12, R6, R14, -R4;                                    /* 0x2008000038631d01 */
                                                                                              /* 0x220042b2c27202d7 */
        /*0c48*/                   DSETP.GT.AND P0, PT, |R8|, |R12|, PT;                      /* 0x1a0e00003081dcc1 */
        /*0c50*/                   SEL R9, R15, R11, P0;                                      /* 0x200000002cf25c04 */
        /*0c58*/                   SEL R8, R14, R10, P0;                                      /* 0x2000000028e21c04 */
        /*0c60*/                   FSETP.GTU.AND P0, PT, |R9|, 1.469367938527859385e-39, PT;  /* 0x260ec0040091dc80 */
        /*0c68*/               @P0 BRK;                                                       /* 0xa8000000000001e7 */
        /*0c70*/                   FSETP.LT.AND P0, PT, |R5|, 1.5046327690525280102e-36, PT;  /* 0x208ec1000051dc80 */
        /*0c78*/                   LOP32I.AND R14, R8, 0xfffffffe;                            /* 0x3bfffffff8839c02 */
                                                                                              /* 0x2202328230427047 */
        /*0c88*/                   MOV32I R11, 0x58500000;                                    /* 0x196140000002dde2 */
        /*0c90*/                   MOV R10, RZ;                                               /* 0x28000000fc029de4 */
        /*0c98*/                   LOP32I.OR R15, R8, 0x1;                                    /* 0x380000000483dc42 */
        /*0ca0*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0ca8*/                   SEL R11, R11, c[0x2][0x4], P0;                             /* 0x2000480010b2dc04 */
        /*0cb0*/                   DMUL R6, R6, R10;                                          /* 0x5000000028619c01 */
        /*0cb8*/                   DMUL R4, R4, R10;                                          /* 0x5000000028411c01 */
                                                                                              /* 0x220282d292328047 */
        /*0cc8*/                   MOV R10, R15;                                              /* 0x280000003c029de4 */
        /*0cd0*/                   MOV R11, R9;                                               /* 0x280000002402dde4 */
        /*0cd8*/                   DFMA R12, R8, R6, -R4;                                     /* 0x2008000018831d01 */
        /*0ce0*/                   DFMA R10, R10, R6, -R4;                                    /* 0x2008000018a29d01 */
        /*0ce8*/                   DSETP.GT.AND P0, PT, |R12|, |R10|, PT;                     /* 0x1a0e000028c1dcc1 */
        /*0cf0*/                   SEL R15, R15, R14, P0;                                     /* 0x2000000038f3dc04 */
        /*0cf8*/                   IADD32I R12.CC, R15, 0x1;                                  /* 0x0c00000004f31c02 */
                                                                                              /* 0x22020042c20042b7 */
        /*0d08*/                   LOP32I.AND R8, R15, 0x1;                                   /* 0x3800000004f21c02 */
        /*0d10*/                   ISETP.EQ.U32.AND P0, PT, R8, 0x1, PT;                      /* 0x190ec0000481dc03 */
        /*0d18*/                   IADD.X R13, RZ, R9;                                        /* 0x4800000027f35c43 */
        /*0d20*/                   IADD32I R10.CC, R15, -0x1;                                 /* 0x0ffffffffcf29c02 */
        /*0d28*/                   SEL R12, R15, R12, !P0;                                    /* 0x2010000030f31c04 */
        /*0d30*/                   IADD32I.X R8, R9, -0x1;                                    /* 0x0bfffffffc921c42 */
        /*0d38*/                   SEL R13, R9, R13, !P0;                                     /* 0x2010000034935c04 */
                                                                                              /* 0x22d2923282020267 */
        /*0d48*/                   SEL R10, R10, R15, !P0;                                    /* 0x201000003ca29c04 */
        /*0d50*/                   SEL R11, R8, R9, !P0;                                      /* 0x201000002482dc04 */
        /*0d58*/                   MOV R8, R12;                                               /* 0x2800000030021de4 */
        /*0d60*/                   MOV R9, R13;                                               /* 0x2800000034025de4 */
        /*0d68*/                   DFMA R8, R6, R8, -R4;                                      /* 0x2008000020621d01 */
        /*0d70*/                   DFMA R4, R6, R10, -R4;                                     /* 0x2008000028611d01 */
        /*0d78*/                   DSETP.GT.AND P0, PT, |R8|, |R4|, PT;                       /* 0x1a0e00001081dcc1 */
                                                                                              /* 0x228042b2d2e04207 */
        /*0d88*/                   SEL R8, R10, R12, P0;                                      /* 0x2000000030a21c04 */
        /*0d90*/                   SEL R9, R11, R13, P0;                                      /* 0x2000000034b25c04 */
        /*0d98*/                   BRK;                                                       /* 0xa800000000001de7 */
        /*0da0*/                   DSETP.EQ.AND P0, PT, R14, RZ, PT;                          /* 0x190e0000fce1dc01 */
        /*0da8*/               @P0 BRA 0xe10;                                                 /* 0x40000001800001e7 */
        /*0db0*/                   MUFU.RCP64H R9, R7;                                        /* 0xc800000018725c00 */
        /*0db8*/                   MOV R8, RZ;                                                /* 0x28000000fc021de4 */
                                                                                              /* 0x220282027202d2d7 */
        /*0dc8*/                   DSETP.GT.AND P1, PT, |R8|, RZ, PT;                         /* 0x1a0e0000fc83dc81 */
        /*0dd0*/              @!P1 DSETP.NEU.AND P0, PT, |R6|, +INF , PT;                     /* 0x1e8edffc0061e481 */
        /*0dd8*/              @!P1 SEL R6, R6, R8, P0;                                        /* 0x200000002061a404 */
        /*0de0*/              @!P1 SEL R7, R7, R9, P0;                                        /* 0x200000002471e404 */
        /*0de8*/              @!P1 MOV R8, R6;                                                /* 0x28000000180225e4 */
        /*0df0*/              @!P1 MOV R9, R7;                                                /* 0x280000001c0265e4 */
        /*0df8*/                   DMUL R8, R4, R8;                                           /* 0x5000000020421c01 */
                                                                                              /* 0x20000002e2b202e7 */
        /*0e08*/                   BRK;                                                       /* 0xa800000000001de7 */
        /*0e10*/                   DMUL R8, R4, R6;                                           /* 0x5000000018421c01 */
        /*0e18*/                   BRK;                                                       /* 0xa800000000001de7 */
        /*0e20*/                   RET;                                                       /* 0x9000000000001de7 */
        /*0e28*/                   BRA 0xe28;                                                 /* 0x4003ffffe0001de7 */
        /*0e30*/                   NOP;                                                       /* 0x4000000000001de4 */
        /*0e38*/                   NOP;                                                       /* 0x4000000000001de4 */
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
        /*00f0*/                   LD.E.64 R4, [R2+0x4e358];                                   /* 0x8400138d60211ca5 */
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
        /*01b0*/                   LD.E.64 R16, [R2+0x5dd98];                                  /* 0x8400177660241ca5 */
        /*01b8*/                   SSY 0x290;                                                  /* 0x6000000340001c07 */
                                                                                               /* 0x22329283f2323237 */
        /*01c8*/                   LD.E.64 R14, [R2+0x2eed8];                                  /* 0x84000bbb60239ca5 */
        /*01d0*/                   LD.E.64 R4, [R2+0x3e918];                                   /* 0x84000fa460211ca5 */
        /*01d8*/                   LD.E.64 R6, [R2+0xfa58];                                    /* 0x840003e960219ca5 */
        /*01e0*/                   LD.E.64 R8, [R2+0x1f498];                                   /* 0x840007d260221ca5 */
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
        /*0250*/                   ST.E.64 [R2+0x2eed8], R12;                                  /* 0x94000bbb60231ca5 */
        /*0258*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0xc], PT;                 /* 0x220e48003001dc80 */
        /*0260*/               @P0 BRA CC.NEU, 0x280;                                          /* 0x40000000600001a7 */
        /*0268*/                   MOV R14, R16;                                               /* 0x2800000040039de4 */
        /*0270*/                   MOV R15, R17;                                               /* 0x280000004403dde4 */
        /*0278*/                   CAL 0x358;                                                  /* 0x5000000360011c07 */
                                                                                               /* 0x23f232e2e04202b7 */
        /*0288*/                   DADD.S R4, R4, R22;                                         /* 0x4800000058411c11 */
        /*0290*/                   DFMA R6, R12, c[0x0][0x148], R6;                            /* 0x200c400520c19c01 */
        /*0298*/                   ST.E.64 [R2+0x3e918], R4;                                   /* 0x94000fa460211ca5 */
        /*02a0*/                   SSY 0x318;                                                  /* 0x60000001c0001c07 */
        /*02a8*/                   ST.E.64 [R2+0xfa58], R6;                                    /* 0x940003e960219ca5 */
        /*02b0*/                   LD.E.64 R12, [R2+0xfa58];                                   /* 0x840003e960231ca5 */
        /*02b8*/                   LD.E.64 R10, [R2+0x3e918];                                  /* 0x84000fa460229ca5 */
                                                                                               /* 0x23f232b2d0429237 */
        /*02c8*/                   DSETP.GT.AND P0, PT, R12, 1, PT;                            /* 0x1a0ecffc00c1dc01 */
        /*02d0*/                   DFMA R8, R10, c[0x0][0x148], R8;                            /* 0x2010400520a21c01 */
        /*02d8*/                   ST.E.64 [R2+0x1f498], R8;                                   /* 0x940007d260221ca5 */
        /*02e0*/                   DSETP.LT.OR P0, PT, R12, -1, P0;                            /* 0x18a0effc00c1dc01 */
        /*02e8*/              @!P0 NOP.S;                                                      /* 0x40000000000021f4 */
        /*02f0*/                   LD.E.64 R4, [R2+0x2eed8];                                   /* 0x84000bbb60211ca5 */
        /*02f8*/                   LD.E.64 R8, [R2+0x1f498];                                   /* 0x840007d260221ca5 */
                                                                                               /* 0x2283f2e2d2d2b287 */
        /*0308*/                   F2F.F64.F64 R4, -R4;                                        /* 0x1000000011b11d04 */
        /*0310*/                   ST.E.64.S [R2+0x2eed8], R4;                                 /* 0x94000bbb60211cb5 */
        /*0318*/                   DSETP.GT.AND P0, PT, R8, 1, PT;                             /* 0x1a0ecffc0081dc01 */
        /*0320*/                   DSETP.LT.OR P0, PT, R8, -1, P0;                             /* 0x18a0effc0081dc01 */
        /*0328*/              @!P0 EXIT;                                                       /* 0x80000000000021e7 */
        /*0330*/                   LD.E.64 R4, [R2+0x3e918];                                   /* 0x84000fa460211ca5 */
        /*0338*/                   F2F.F64.F64 R4, -R4;                                        /* 0x1000000011b11d04 */
                                                                                               /* 0x22b200428042e047 */
        /*0348*/                   ST.E.64 [R2+0x3e918], R4;                                   /* 0x94000fa460211ca5 */
        /*0350*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*0358*/                   LOP32I.AND R0, R11, 0x40000000;                             /* 0x3900000000b01c02 */
        /*0360*/                   LOP32I.AND R26, R15, 0x7f800000;                            /* 0x39fe000000f69c02 */
        /*0368*/                   ISETP.LT.U32.AND P0, PT, R0, c[0x2][0x10], PT;              /* 0x188e48004001dc03 */
        /*0370*/                   MOV32I R0, 0x5ff00000;                                      /* 0x197fc00000001de2 */
        /*0378*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
                                                                                               /* 0x2280428042028047 */
        /*0388*/                   SEL R17, R0, c[0x2][0x14], P0;                              /* 0x2000480050045c04 */
        /*0390*/                   MOV32I R20, 0x1;                                            /* 0x1800000004051de2 */
        /*0398*/                   DMUL R18, R10, R16;                                         /* 0x5000000040a49c01 */
        /*03a0*/                   ISETP.LT.U32.AND P0, PT, R26, c[0x2][0x18], PT;             /* 0x188e480061a1dc03 */
        /*03a8*/                   MOV R24, RZ;                                                /* 0x28000000fc061de4 */
        /*03b0*/                   MUFU.RCP64H R21, R19;                                       /* 0xc800000019355c00 */
        /*03b8*/                   PBK 0x650;                                                  /* 0x6800000a40001c07 */
                                                                                               /* 0x2292929252328207 */
        /*03c8*/                   DFMA R22, -R18, R20, c[0x2][0x0];                           /* 0x2028880001259e01 */
        /*03d0*/                   SEL R25, R0, c[0x2][0x14], P0;                              /* 0x2000480050065c04 */
        /*03d8*/                   DFMA R30, R22, R22, R22;                                    /* 0x202c000059679c01 */
        /*03e0*/                   DMUL R22, R14, R24;                                         /* 0x5000000060e59c01 */
        /*03e8*/                   DFMA R20, R20, R30, R20;                                    /* 0x2028000079451c01 */
        /*03f0*/                   DMUL R24, R22, R20;                                         /* 0x5000000051661c01 */
        /*03f8*/                   DFMA R18, -R18, R24, R22;                                   /* 0x202c000061249e01 */
                                                                                               /* 0x2282823042b2d297 */
        /*0408*/                   DFMA R22, R20, R18, R24;                                    /* 0x2030000049459c01 */
        /*0410*/                   DSETP.LEU.AND P0, PT, |R22|, RZ, PT;                        /* 0x1d8e0000fd61dc81 */
        /*0418*/               @P0 BRA 0x5d0;                                                  /* 0x40000006c00001e7 */
        /*0420*/                   ISETP.GT.U32.AND P0, PT, R26, c[0x2][0x1c], PT;             /* 0x1a0e480071a1dc03 */
        /*0428*/                   MOV R18, RZ;                                                /* 0x28000000fc049de4 */
        /*0430*/                   DMUL R20, R16, R22;                                         /* 0x5000000059051c01 */
        /*0438*/                   SEL R19, R0, c[0x2][0x14], P0;                              /* 0x200048005004dc04 */
                                                                                               /* 0x2202d29252325237 */
        /*0448*/                   DMUL R22, R22, R18;                                         /* 0x5000000049659c01 */
        /*0450*/                   DMUL R20, R18, R20;                                         /* 0x5000000051251c01 */
        /*0458*/                   DMUL R16, R16, R22;                                         /* 0x5000000059041c01 */
        /*0460*/                   DFMA R18, R10, R20, -R14;                                   /* 0x201c000050a49d01 */
        /*0468*/                   DFMA R22, R10, R16, -R14;                                   /* 0x201c000040a59d01 */
        /*0470*/                   DSETP.GT.AND P0, PT, |R18|, |R22|, PT;                      /* 0x1a0e00005921dcc1 */
        /*0478*/                   SEL R17, R17, R21, P0;                                      /* 0x2000000055145c04 */
                                                                                               /* 0x22220042b042c277 */
        /*0488*/                   SEL R18, R16, R20, P0;                                      /* 0x2000000051049c04 */
        /*0490*/                   FSETP.GTU.AND P0, PT, |R17|, 1.469367938527859385e-39, PT;  /* 0x260ec0040111dc80 */
        /*0498*/                   MOV R19, R17;                                               /* 0x280000004404dde4 */
        /*04a0*/               @P0 BRK;                                                        /* 0xa8000000000001e7 */
        /*04a8*/                   FSETP.LT.AND P0, PT, |R15|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000f1dc80 */
        /*04b0*/                   LOP32I.OR R22, R18, 0x1;                                    /* 0x3800000005259c42 */
        /*04b8*/                   MOV32I R21, 0x58500000;                                     /* 0x1961400000055de2 */
                                                                                               /* 0x2272020232828047 */
        /*04c8*/                   MOV R20, RZ;                                                /* 0x28000000fc051de4 */
        /*04d0*/                   LOP32I.AND R16, R18, 0xfffffffe;                            /* 0x3bfffffff9241c02 */
        /*04d8*/                   SEL R21, R21, c[0x2][0x4], P0;                              /* 0x2000480011555c04 */
        /*04e0*/                   DMUL R10, R10, R20;                                         /* 0x5000000050a29c01 */
        /*04e8*/                   DMUL R14, R14, R20;                                         /* 0x5000000050e39c01 */
        /*04f0*/                   MOV R20, R22;                                               /* 0x2800000058051de4 */
        /*04f8*/                   MOV R21, R17;                                               /* 0x2800000044055de4 */
                                                                                               /* 0x2232804282d29237 */
        /*0508*/                   DFMA R18, R16, R10, -R14;                                   /* 0x201c000029049d01 */
        /*0510*/                   DFMA R20, R20, R10, -R14;                                   /* 0x201c000029451d01 */
        /*0518*/                   DSETP.GT.AND P0, PT, |R18|, |R20|, PT;                      /* 0x1a0e00005121dcc1 */
        /*0520*/                   SEL R23, R22, R16, P0;                                      /* 0x200000004165dc04 */
        /*0528*/                   IADD32I R18.CC, R23, 0x1;                                   /* 0x0c00000005749c02 */
        /*0530*/                   LOP32I.AND R16, R23, 0x1;                                   /* 0x3800000005741c02 */
        /*0538*/                   ISETP.EQ.U32.AND P0, PT, R16, 0x1, PT;                      /* 0x190ec0000501dc03 */
                                                                                               /* 0x220262020042c047 */
        /*0548*/                   IADD.X R0, RZ, R17;                                         /* 0x4800000047f01c43 */
        /*0550*/                   IADD32I R20.CC, R23, -0x1;                                  /* 0x0ffffffffd751c02 */
        /*0558*/                   SEL R18, R23, R18, !P0;                                     /* 0x2010000049749c04 */
        /*0560*/                   IADD32I.X R21, R17, -0x1;                                   /* 0x0bfffffffd155c42 */
        /*0568*/                   SEL R19, R17, R0, !P0;                                      /* 0x201000000114dc04 */
        /*0570*/                   SEL R20, R20, R23, !P0;                                     /* 0x201000005d451c04 */
        /*0578*/                   SEL R21, R21, R17, !P0;                                     /* 0x2010000045555c04 */
                                                                                               /* 0x220202d292328047 */
        /*0588*/                   MOV R16, R18;                                               /* 0x2800000048041de4 */
        /*0590*/                   MOV R17, R19;                                               /* 0x280000004c045de4 */
        /*0598*/                   DFMA R16, R10, R16, -R14;                                   /* 0x201c000040a41d01 */
        /*05a0*/                   DFMA R10, R10, R20, -R14;                                   /* 0x201c000050a29d01 */
        /*05a8*/                   DSETP.GT.AND P0, PT, |R16|, |R10|, PT;                      /* 0x1a0e00002901dcc1 */
        /*05b0*/                   SEL R18, R20, R18, P0;                                      /* 0x2000000049449c04 */
        /*05b8*/                   SEL R19, R21, R19, P0;                                      /* 0x200000004d54dc04 */
                                                                                               /* 0x22d2d28042b2d2e7 */
        /*05c8*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*05d0*/                   DSETP.EQ.AND P0, PT, R22, RZ, PT;                           /* 0x190e0000fd61dc01 */
        /*05d8*/               @P0 BRA 0x638;                                                  /* 0x40000001600001e7 */
        /*05e0*/                   MUFU.RCP64H R17, R11;                                       /* 0xc800000018b45c00 */
        /*05e8*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*05f0*/                   DSETP.GT.AND P1, PT, |R16|, RZ, PT;                         /* 0x1a0e0000fd03dc81 */
        /*05f8*/              @!P1 DSETP.NEU.AND P0, PT, |R10|, +INF , PT;                     /* 0x1e8edffc00a1e481 */
                                                                                               /* 0x2202e20282027207 */
        /*0608*/              @!P1 SEL R0, R10, R16, P0;                                       /* 0x2000000040a02404 */
        /*0610*/              @!P1 SEL R11, R11, R17, P0;                                      /* 0x2000000044b2e404 */
        /*0618*/              @!P1 MOV R16, R0;                                                /* 0x28000000000425e4 */
        /*0620*/              @!P1 MOV R17, R11;                                               /* 0x280000002c0465e4 */
        /*0628*/                   DMUL R18, R14, R16;                                         /* 0x5000000040e49c01 */
        /*0630*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*0638*/                   DMUL R18, R14, R10;                                         /* 0x5000000028e49c01 */
                                                                                               /* 0x20000002e20042b7 */
        /*0648*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*0650*/                   MOV R22, R18;                                               /* 0x2800000048059de4 */
        /*0658*/                   MOV R23, R19;                                               /* 0x280000004c05dde4 */
        /*0660*/                   RET;                                                        /* 0x9000000000001de7 */
        /*0668*/                   BRA 0x668;                                                  /* 0x4003ffffe0001de7 */
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
        /*0090*/                   MOV32I R44, 0x1f40;                                         /* 0x1800007d000b1de2 */
        /*0098*/                   SHR.U32 R3, R37, 0x1d;                                      /* 0x5800c0007650dc03 */
        /*00a0*/                   MOV R43, RZ;                                                /* 0x28000000fc0adde4 */
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
                                                                                               /* 0x22028042b2c2c047 */
        /*0148*/                   ISUB RZ.CC, R32, R33;                                       /* 0x48010000860fdd03 */
        /*0150*/                   SSY 0x5c0;                                                  /* 0x60000011a0001c07 */
        /*0158*/                   ISETP.EQ.X.AND P0, PT, R42, R36, PT;                        /* 0x190e000092a1dc63 */
        /*0160*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*0168*/                   MOV32I R9, 0x3;                                             /* 0x180000000c025de2 */
        /*0170*/                   SHL R12, R32, -0x1d;                                        /* 0x6000ffff8e031c03 */
        /*0178*/                   IADD32I RZ.CC, R9, -0x20;                                   /* 0x0fffffff809fdc02 */
                                                                                               /* 0x220042c23042b047 */
        /*0188*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
        /*0190*/                   SHL R11, R42, 0x3;                                          /* 0x6000c0000ea2dc03 */
        /*0198*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*01a0*/                   SSY 0x318;                                                  /* 0x60000005c0001c07 */
        /*01a8*/                   ISCADD R9.CC, R37, 0x8, 0x3;                                /* 0x4001c00022525c63 */
        /*01b0*/                   IMAD.HI.X R10, R37, 0x8, RZ;                                /* 0x20fec00022529ce3 */
        /*01b8*/                   IADD R16.CC, R9, R18;                                       /* 0x4801000048941c03 */
                                                                                               /* 0x228042c04202b047 */
        /*01c8*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*01d0*/                   SHR.U32 R13, R32, 0x1d;                                     /* 0x5800c00076035c03 */
        /*01d8*/                   IADD.X R17, R10, R9;                                        /* 0x4800000024a45c43 */
        /*01e0*/                   IMAD.U32.U32 R20.CC, R32, 0x8, R18;                         /* 0x2025c00022051c03 */
        /*01e8*/               @P0 IADD R12, R11, R13;                                         /* 0x4800000034b30003 */
        /*01f0*/                   IADD.X R21, R12, R9;                                        /* 0x4800000024c55c43 */
        /*01f8*/                   LD.E.64 R24, [R16+0x1f490];                                 /* 0x840007d241061ca5 */
                                                                                               /* 0x223273f042a23237 */
        /*0208*/                   LD.E.64 R10, [R20+0x1f490];                                 /* 0x840007d241429ca5 */
        /*0210*/                   LD.E.64 R26, [R16+0xfa50];                                  /* 0x840003e941069ca5 */
        /*0218*/                   LD.E.64 R12, [R20+0xfa50];                                  /* 0x840003e941431ca5 */
        /*0220*/                   MOV32I R16, 0x0;                                            /* 0x1800000000041de2 */
        /*0228*/                   MOV32I R17, 0x3fe00000;                                     /* 0x18ff800000045de2 */
        /*0230*/                   DADD R24, -R24, R10;                                        /* 0x4800000029861e01 */
        /*0238*/                   DADD R26, -R26, R12;                                        /* 0x4800000031a69e01 */
                                                                                               /* 0x2292928280429297 */
        /*0248*/                   DMUL R12, R24, R24;                                         /* 0x5000000061831c01 */
        /*0250*/                   DFMA R10, R26, R26, R12;                                    /* 0x2018000069a29c01 */
        /*0258*/                   IADD32I R19, R11, -0x3500000;                               /* 0x0bf2c00000b4dc02 */
        /*0260*/                   MUFU.RSQ64H R13, R11;                                       /* 0xc80000001cb35c00 */
        /*0268*/                   MOV R12, R19;                                               /* 0x280000004c031de4 */
        /*0270*/                   DMUL R14, R12, R12;                                         /* 0x5000000030c39c01 */
        /*0278*/                   DFMA R14, -R10, R14, c[0x2][0x0];                           /* 0x201c880000a39e01 */
                                                                                               /* 0x2280420282029237 */
        /*0288*/                   DFMA R16, R14, 0.375, R16;                                  /* 0x2020cff600e41c01 */
        /*0290*/                   DMUL R14, R12, R14;                                         /* 0x5000000038c39c01 */
        /*0298*/                   DFMA R20, R16, R14, R12;                                    /* 0x2018000039051c01 */
        /*02a0*/                   ISETP.LT.U32.AND P0, PT, R19, c[0x2][0x8], PT;              /* 0x188e48002131dc03 */
        /*02a8*/                   DMUL R16, R10, R20;                                         /* 0x5000000050a41c01 */
        /*02b0*/                   IADD32I R23, R21, -0x100000;                                /* 0x0bffc0000155dc02 */
        /*02b8*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
                                                                                               /* 0x22b20042b2028207 */
        /*02c8*/                   DFMA R12, -R16, R16, R10;                                   /* 0x2014000041031e01 */
        /*02d0*/                   MOV R15, R23;                                               /* 0x280000005c03dde4 */
        /*02d8*/                   DFMA R28, R12, R14, R16;                                    /* 0x2020000038c71c01 */
        /*02e0*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*02e8*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*02f0*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*02f8*/                   MOV R19, R23;                                               /* 0x280000005c04dde4 */
                                                                                               /* 0x22028042c042b2b7 */
        /*0308*/                   CAL 0x1bd0;                                                 /* 0x5000006300011c07 */
        /*0310*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0318*/                   IMAD R20.CC, R37, 0x8, R18;                                 /* 0x2025c00022551ca3 */
        /*0320*/                   LD.E.64 R12, [R4+0x18];                                     /* 0x8400000060431ca5 */
        /*0328*/                   IMAD.HI.X R21, R37, 0x8, R9;                                /* 0x2092c00022555ce3 */
        /*0330*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*0338*/                   LD.E.64 R10, [R20+0x18];                                    /* 0x8400000061429ca5 */
                                                                                               /* 0x2293f29292804297 */
        /*0348*/                   DFMA R16, R28, R28, c[0x2][0x18];                           /* 0x2038880061c41c01 */
        /*0350*/                   MUFU.RCP64H R15, R17;                                       /* 0xc80000001913dc00 */
        /*0358*/                   SSY 0x418;                                                  /* 0x60000002e0001c07 */
        /*0360*/                   DFMA R18, -R16, R14, c[0x2][0x0];                           /* 0x201c880001049e01 */
        /*0368*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*0370*/                   DFMA R14, R14, R18, R14;                                    /* 0x201c000048e39c01 */
        /*0378*/                   DMUL R10, R10, c[0x2][0x10];                                /* 0x5000480040a29c01 */
                                                                                               /* 0x22c2828202929297 */
        /*0388*/                   DMUL R52, R10, R12;                                         /* 0x5000000030ad1c01 */
        /*0390*/                   DMUL R10, R52, R14;                                         /* 0x500000003b429c01 */
        /*0398*/                   DFMA R12, -R16, R10, R52;                                   /* 0x2068000029031e01 */
        /*03a0*/                   DFMA R30, R14, R12, R10;                                    /* 0x2014000030e79c01 */
        /*03a8*/                   FSET.GTU.AND RZ.CC, |R53|, c[0x2][0xc], PT;                 /* 0x160f4800335fdc80 */
        /*03b0*/                   FFMA R9, RZ, R17, R31;                                      /* 0x303e000047f25c00 */
        /*03b8*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
                                                                                               /* 0x2202b2b0420042b7 */
        /*03c8*/               @P0 BRA CC.NEU, 0x410;                                          /* 0x40000001000001a7 */
        /*03d0*/                   MOV R10, R52;                                               /* 0x28000000d0029de4 */
        /*03d8*/                   MOV R11, R53;                                               /* 0x28000000d402dde4 */
        /*03e0*/                   MOV R9, R16;                                                /* 0x2800000040025de4 */
        /*03e8*/                   MOV R12, R17;                                               /* 0x2800000044031de4 */
        /*03f0*/                   CAL 0x1868;                                                 /* 0x50000051c0011c07 */
        /*03f8*/                   MOV R30, R16;                                               /* 0x2800000040079de4 */
                                                                                               /* 0x229292928042b2b7 */
        /*0408*/                   MOV R31, R17;                                               /* 0x280000004407dde4 */
        /*0410*/                   MUFU.RCP64H.S R11, R29;                                     /* 0xc800000019d2dc10 */
        /*0418*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*0420*/                   DMUL R14, R26, R30;                                         /* 0x5000000079a39c01 */
        /*0428*/                   DFMA R12, -R28, R10, c[0x2][0x0];                           /* 0x2014880001c31e01 */
        /*0430*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*0438*/                   DFMA R10, R10, R12, R10;                                    /* 0x2014000030a29c01 */
                                                                                               /* 0x22c2828042029297 */
        /*0448*/                   DMUL R16, R14, R10;                                         /* 0x5000000028e41c01 */
        /*0450*/                   DFMA R12, -R28, R16, R14;                                   /* 0x201c000041c31e01 */
        /*0458*/                   DFMA R16, R10, R12, R16;                                    /* 0x2020000030a41c01 */
        /*0460*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0xc], PT;                 /* 0x160f480030ffdc80 */
        /*0468*/                   SSY 0x4c0;                                                  /* 0x6000000140001c07 */
        /*0470*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*0478*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
                                                                                               /* 0x22b2b2b0420042b7 */
        /*0488*/               @P0 BRA CC.NEU, 0x4b8;                                          /* 0x40000000a00001a7 */
        /*0490*/                   MOV R10, R14;                                               /* 0x2800000038029de4 */
        /*0498*/                   MOV R11, R15;                                               /* 0x280000003c02dde4 */
        /*04a0*/                   MOV R9, R28;                                                /* 0x2800000070025de4 */
        /*04a8*/                   MOV R12, R29;                                               /* 0x2800000074031de4 */
        /*04b0*/                   CAL 0x1868;                                                 /* 0x5000004ec0011c07 */
        /*04b8*/                   NOP.S;                                                      /* 0x4000000000001df4 */
                                                                                               /* 0x2292929280420047 */
        /*04c8*/                   LD.E.64 R10, [R2+0x4e358];                                  /* 0x8400138d60229ca5 */
        /*04d0*/                   MUFU.RCP64H R15, R29;                                       /* 0xc800000019d3dc00 */
        /*04d8*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*04e0*/                   DMUL R24, R24, R30;                                         /* 0x5000000079861c01 */
        /*04e8*/                   DFMA R12, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c31e01 */
        /*04f0*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*04f8*/                   DFMA R18, R14, R12, R14;                                    /* 0x201c000030e49c01 */
                                                                                               /* 0x22829233f0420297 */
        /*0508*/                   DMUL R14, R18, R24;                                         /* 0x5000000061239c01 */
        /*0510*/                   DFMA R12, -R28, R14, R24;                                   /* 0x2030000039c31e01 */
        /*0518*/                   FSET.GTU.AND RZ.CC, |R25|, c[0x2][0xc], PT;                 /* 0x160f4800319fdc80 */
        /*0520*/                   SSY 0x5a8;                                                  /* 0x6000000200001c07 */
        /*0528*/                   DADD R16, R16, R10;                                         /* 0x4800000029041c01 */
        /*0530*/                   DFMA R10, R18, R12, R14;                                    /* 0x201c000031229c01 */
        /*0538*/                   FFMA R9, RZ, R29, R11;                                      /* 0x3016000077f25c00 */
                                                                                               /* 0x22b0420042b2c047 */
        /*0548*/                   ST.E.64 [R2+0x4e358], R16;                                  /* 0x9400138d60241ca5 */
        /*0550*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*0558*/               @P0 BRA CC.NEU, 0x5a0;                                          /* 0x40000001000001a7 */
        /*0560*/                   MOV R10, R24;                                               /* 0x2800000060029de4 */
        /*0568*/                   MOV R11, R25;                                               /* 0x280000006402dde4 */
        /*0570*/                   MOV R9, R28;                                                /* 0x2800000070025de4 */
        /*0578*/                   MOV R12, R29;                                               /* 0x2800000074031de4 */
                                                                                               /* 0x22b293f2b2b042b7 */
        /*0588*/                   CAL 0x1868;                                                 /* 0x5000004b60011c07 */
        /*0590*/                   MOV R10, R16;                                               /* 0x2800000040029de4 */
        /*0598*/                   MOV R11, R17;                                               /* 0x280000004402dde4 */
        /*05a0*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*05a8*/                   LD.E.64 R12, [R2+0x5dd98];                                  /* 0x8400177660231ca5 */
        /*05b0*/                   DADD R12, R10, R12;                                         /* 0x4800000030a31c01 */
        /*05b8*/                   ST.E.64.S [R2+0x5dd98], R12;                                /* 0x9400177660231cb5 */
                                                                                               /* 0x22028042b2c2c047 */
        /*05c8*/                   ISUB RZ.CC, R0, R33;                                        /* 0x48010000840fdd03 */
        /*05d0*/                   SSY 0xa40;                                                  /* 0x60000011a0001c07 */
        /*05d8*/                   ISETP.EQ.X.AND P0, PT, R41, R36, PT;                        /* 0x190e00009291dc63 */
        /*05e0*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*05e8*/                   MOV32I R9, 0x3;                                             /* 0x180000000c025de2 */
        /*05f0*/                   SHL R10, R0, -0x1d;                                         /* 0x6000ffff8c029c03 */
        /*05f8*/                   IADD32I RZ.CC, R9, -0x20;                                   /* 0x0fffffff809fdc02 */
                                                                                               /* 0x220042c23042b047 */
        /*0608*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
        /*0610*/                   SHL R11, R41, 0x3;                                          /* 0x6000c0000e92dc03 */
        /*0618*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*0620*/                   SSY 0x798;                                                  /* 0x60000005c0001c07 */
        /*0628*/                   ISCADD R9.CC, R37, 0x8, 0x3;                                /* 0x4001c00022525c63 */
        /*0630*/                   IMAD.HI.X R13, R37, 0x8, RZ;                                /* 0x20fec00022535ce3 */
        /*0638*/                   IADD R20.CC, R9, R18;                                       /* 0x4801000048951c03 */
                                                                                               /* 0x228280428042b047 */
        /*0648*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*0650*/                   SHR.U32 R12, R0, 0x1d;                                      /* 0x5800c00074031c03 */
        /*0658*/                   IADD.X R21, R13, R9;                                        /* 0x4800000024d55c43 */
        /*0660*/                   IMAD.U32.U32 R16.CC, R0, 0x8, R18;                          /* 0x2025c00020041c03 */
        /*0668*/               @P0 IADD R10, R11, R12;                                         /* 0x4800000030b28003 */
        /*0670*/                   LD.E.64 R24, [R20+0x1f490];                                 /* 0x840007d241461ca5 */
        /*0678*/                   IADD.X R17, R10, R9;                                        /* 0x4800000024a45c43 */
                                                                                               /* 0x223273f042a23237 */
        /*0688*/                   LD.E.64 R10, [R16+0x1f490];                                 /* 0x840007d241029ca5 */
        /*0690*/                   LD.E.64 R26, [R20+0xfa50];                                  /* 0x840003e941469ca5 */
        /*0698*/                   LD.E.64 R12, [R16+0xfa50];                                  /* 0x840003e941031ca5 */
        /*06a0*/                   MOV32I R20, 0x0;                                            /* 0x1800000000051de2 */
        /*06a8*/                   MOV32I R21, 0x3fe00000;                                     /* 0x18ff800000055de2 */
        /*06b0*/                   DADD R24, -R24, R10;                                        /* 0x4800000029861e01 */
        /*06b8*/                   DADD R26, -R26, R12;                                        /* 0x4800000031a69e01 */
                                                                                               /* 0x2292928280429297 */
        /*06c8*/                   DMUL R12, R24, R24;                                         /* 0x5000000061831c01 */
        /*06d0*/                   DFMA R10, R26, R26, R12;                                    /* 0x2018000069a29c01 */
        /*06d8*/                   IADD32I R19, R11, -0x3500000;                               /* 0x0bf2c00000b4dc02 */
        /*06e0*/                   MUFU.RSQ64H R13, R11;                                       /* 0xc80000001cb35c00 */
        /*06e8*/                   MOV R12, R19;                                               /* 0x280000004c031de4 */
        /*06f0*/                   DMUL R14, R12, R12;                                         /* 0x5000000030c39c01 */
        /*06f8*/                   DFMA R14, -R10, R14, c[0x2][0x0];                           /* 0x201c880000a39e01 */
                                                                                               /* 0x2280420282029237 */
        /*0708*/                   DFMA R16, R14, 0.375, R20;                                  /* 0x2028cff600e41c01 */
        /*0710*/                   DMUL R14, R12, R14;                                         /* 0x5000000038c39c01 */
        /*0718*/                   DFMA R20, R16, R14, R12;                                    /* 0x2018000039051c01 */
        /*0720*/                   ISETP.LT.U32.AND P0, PT, R19, c[0x2][0x8], PT;              /* 0x188e48002131dc03 */
        /*0728*/                   DMUL R16, R10, R20;                                         /* 0x5000000050a41c01 */
        /*0730*/                   IADD32I R23, R21, -0x100000;                                /* 0x0bffc0000155dc02 */
        /*0738*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
                                                                                               /* 0x22b20042b2028207 */
        /*0748*/                   DFMA R12, -R16, R16, R10;                                   /* 0x2014000041031e01 */
        /*0750*/                   MOV R15, R23;                                               /* 0x280000005c03dde4 */
        /*0758*/                   DFMA R28, R12, R14, R16;                                    /* 0x2020000038c71c01 */
        /*0760*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*0768*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*0770*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*0778*/                   MOV R19, R23;                                               /* 0x280000005c04dde4 */
                                                                                               /* 0x22028042c042b2b7 */
        /*0788*/                   CAL 0x1bd0;                                                 /* 0x5000005100011c07 */
        /*0790*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0798*/                   IMAD R22.CC, R37, 0x8, R18;                                 /* 0x2025c00022559ca3 */
        /*07a0*/                   LD.E.64 R12, [R4+0x20];                                     /* 0x8400000080431ca5 */
        /*07a8*/                   IMAD.HI.X R23, R37, 0x8, R9;                                /* 0x2092c0002255dce3 */
        /*07b0*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*07b8*/                   LD.E.64 R10, [R22+0x18];                                    /* 0x8400000061629ca5 */
                                                                                               /* 0x2293f29292804297 */
        /*07c8*/                   DFMA R16, R28, R28, c[0x2][0x18];                           /* 0x2038880061c41c01 */
        /*07d0*/                   MUFU.RCP64H R15, R17;                                       /* 0xc80000001913dc00 */
        /*07d8*/                   SSY 0x898;                                                  /* 0x60000002e0001c07 */
        /*07e0*/                   DFMA R18, -R16, R14, c[0x2][0x0];                           /* 0x201c880001049e01 */
        /*07e8*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*07f0*/                   DFMA R14, R14, R18, R14;                                    /* 0x201c000048e39c01 */
        /*07f8*/                   DMUL R10, R10, c[0x2][0x10];                                /* 0x5000480040a29c01 */
                                                                                               /* 0x22c2828202929297 */
        /*0808*/                   DMUL R20, R10, R12;                                         /* 0x5000000030a51c01 */
        /*0810*/                   DMUL R10, R20, R14;                                         /* 0x5000000039429c01 */
        /*0818*/                   DFMA R12, -R16, R10, R20;                                   /* 0x2028000029031e01 */
        /*0820*/                   DFMA R30, R14, R12, R10;                                    /* 0x2014000030e79c01 */
        /*0828*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
        /*0830*/                   FFMA R9, RZ, R17, R31;                                      /* 0x303e000047f25c00 */
        /*0838*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
                                                                                               /* 0x2202b2b0420042b7 */
        /*0848*/               @P0 BRA CC.NEU, 0x890;                                          /* 0x40000001000001a7 */
        /*0850*/                   MOV R10, R20;                                               /* 0x2800000050029de4 */
        /*0858*/                   MOV R11, R21;                                               /* 0x280000005402dde4 */
        /*0860*/                   MOV R9, R16;                                                /* 0x2800000040025de4 */
        /*0868*/                   MOV R12, R17;                                               /* 0x2800000044031de4 */
        /*0870*/                   CAL 0x1868;                                                 /* 0x5000003fc0011c07 */
        /*0878*/                   MOV R30, R16;                                               /* 0x2800000040079de4 */
                                                                                               /* 0x229292928042b2b7 */
        /*0888*/                   MOV R31, R17;                                               /* 0x280000004407dde4 */
        /*0890*/                   MUFU.RCP64H.S R11, R29;                                     /* 0xc800000019d2dc10 */
        /*0898*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*08a0*/                   DMUL R14, R26, R30;                                         /* 0x5000000079a39c01 */
        /*08a8*/                   DFMA R12, -R28, R10, c[0x2][0x0];                           /* 0x2014880001c31e01 */
        /*08b0*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*08b8*/                   DFMA R10, R10, R12, R10;                                    /* 0x2014000030a29c01 */
                                                                                               /* 0x22c2828042029297 */
        /*08c8*/                   DMUL R16, R14, R10;                                         /* 0x5000000028e41c01 */
        /*08d0*/                   DFMA R12, -R28, R16, R14;                                   /* 0x201c000041c31e01 */
        /*08d8*/                   DFMA R16, R10, R12, R16;                                    /* 0x2020000030a41c01 */
        /*08e0*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0xc], PT;                 /* 0x160f480030ffdc80 */
        /*08e8*/                   SSY 0x940;                                                  /* 0x6000000140001c07 */
        /*08f0*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*08f8*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
                                                                                               /* 0x22b2b2b0420042b7 */
        /*0908*/               @P0 BRA CC.NEU, 0x938;                                          /* 0x40000000a00001a7 */
        /*0910*/                   MOV R10, R14;                                               /* 0x2800000038029de4 */
        /*0918*/                   MOV R11, R15;                                               /* 0x280000003c02dde4 */
        /*0920*/                   MOV R9, R28;                                                /* 0x2800000070025de4 */
        /*0928*/                   MOV R12, R29;                                               /* 0x2800000074031de4 */
        /*0930*/                   CAL 0x1868;                                                 /* 0x5000003cc0011c07 */
        /*0938*/                   NOP.S;                                                      /* 0x4000000000001df4 */
                                                                                               /* 0x2292929280420047 */
        /*0948*/                   LD.E.64 R10, [R2+0x4e358];                                  /* 0x8400138d60229ca5 */
        /*0950*/                   MUFU.RCP64H R15, R29;                                       /* 0xc800000019d3dc00 */
        /*0958*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*0960*/                   DMUL R24, R24, R30;                                         /* 0x5000000079861c01 */
        /*0968*/                   DFMA R12, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c31e01 */
        /*0970*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*0978*/                   DFMA R18, R14, R12, R14;                                    /* 0x201c000030e49c01 */
                                                                                               /* 0x22829233f0420297 */
        /*0988*/                   DMUL R14, R18, R24;                                         /* 0x5000000061239c01 */
        /*0990*/                   DFMA R12, -R28, R14, R24;                                   /* 0x2030000039c31e01 */
        /*0998*/                   FSET.GTU.AND RZ.CC, |R25|, c[0x2][0xc], PT;                 /* 0x160f4800319fdc80 */
        /*09a0*/                   SSY 0xa28;                                                  /* 0x6000000200001c07 */
        /*09a8*/                   DADD R16, R16, R10;                                         /* 0x4800000029041c01 */
        /*09b0*/                   DFMA R10, R18, R12, R14;                                    /* 0x201c000031229c01 */
        /*09b8*/                   FFMA R9, RZ, R29, R11;                                      /* 0x3016000077f25c00 */
                                                                                               /* 0x22b0420042b2c047 */
        /*09c8*/                   ST.E.64 [R2+0x4e358], R16;                                  /* 0x9400138d60241ca5 */
        /*09d0*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*09d8*/               @P0 BRA CC.NEU, 0xa20;                                          /* 0x40000001000001a7 */
        /*09e0*/                   MOV R10, R24;                                               /* 0x2800000060029de4 */
        /*09e8*/                   MOV R11, R25;                                               /* 0x280000006402dde4 */
        /*09f0*/                   MOV R9, R28;                                                /* 0x2800000070025de4 */
        /*09f8*/                   MOV R12, R29;                                               /* 0x2800000074031de4 */
                                                                                               /* 0x22b293f2b2b042b7 */
        /*0a08*/                   CAL 0x1868;                                                 /* 0x5000003960011c07 */
        /*0a10*/                   MOV R10, R16;                                               /* 0x2800000040029de4 */
        /*0a18*/                   MOV R11, R17;                                               /* 0x280000004402dde4 */
        /*0a20*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0a28*/                   LD.E.64 R12, [R2+0x5dd98];                                  /* 0x8400177660231ca5 */
        /*0a30*/                   DADD R12, R10, R12;                                         /* 0x4800000030a31c01 */
        /*0a38*/                   ST.E.64.S [R2+0x5dd98], R12;                                /* 0x9400177660231cb5 */
                                                                                               /* 0x22028042b2c2c047 */
        /*0a48*/                   ISUB RZ.CC, R6, R33;                                        /* 0x48010000846fdd03 */
        /*0a50*/                   SSY 0xec0;                                                  /* 0x60000011a0001c07 */
        /*0a58*/                   ISETP.EQ.X.AND P0, PT, R40, R36, PT;                        /* 0x190e00009281dc63 */
        /*0a60*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*0a68*/                   MOV32I R9, 0x3;                                             /* 0x180000000c025de2 */
        /*0a70*/                   SHL R10, R6, -0x1d;                                         /* 0x6000ffff8c629c03 */
        /*0a78*/                   IADD32I RZ.CC, R9, -0x20;                                   /* 0x0fffffff809fdc02 */
                                                                                               /* 0x220042c23042b047 */
        /*0a88*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
        /*0a90*/                   SHL R11, R40, 0x3;                                          /* 0x6000c0000e82dc03 */
        /*0a98*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*0aa0*/                   SSY 0xc18;                                                  /* 0x60000005c0001c07 */
        /*0aa8*/                   ISCADD R9.CC, R37, 0x8, 0x3;                                /* 0x4001c00022525c63 */
        /*0ab0*/                   IMAD.HI.X R13, R37, 0x8, RZ;                                /* 0x20fec00022535ce3 */
        /*0ab8*/                   IADD R20.CC, R9, R18;                                       /* 0x4801000048951c03 */
                                                                                               /* 0x228280428042b047 */
        /*0ac8*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*0ad0*/                   SHR.U32 R12, R6, 0x1d;                                      /* 0x5800c00074631c03 */
        /*0ad8*/                   IADD.X R21, R13, R9;                                        /* 0x4800000024d55c43 */
        /*0ae0*/                   IMAD.U32.U32 R16.CC, R6, 0x8, R18;                          /* 0x2025c00020641c03 */
        /*0ae8*/               @P0 IADD R10, R11, R12;                                         /* 0x4800000030b28003 */
        /*0af0*/                   LD.E.64 R24, [R20+0x1f490];                                 /* 0x840007d241461ca5 */
        /*0af8*/                   IADD.X R17, R10, R9;                                        /* 0x4800000024a45c43 */
                                                                                               /* 0x223273f042a23237 */
        /*0b08*/                   LD.E.64 R10, [R16+0x1f490];                                 /* 0x840007d241029ca5 */
        /*0b10*/                   LD.E.64 R26, [R20+0xfa50];                                  /* 0x840003e941469ca5 */
        /*0b18*/                   LD.E.64 R12, [R16+0xfa50];                                  /* 0x840003e941031ca5 */
        /*0b20*/                   MOV32I R20, 0x0;                                            /* 0x1800000000051de2 */
        /*0b28*/                   MOV32I R21, 0x3fe00000;                                     /* 0x18ff800000055de2 */
        /*0b30*/                   DADD R24, -R24, R10;                                        /* 0x4800000029861e01 */
        /*0b38*/                   DADD R26, -R26, R12;                                        /* 0x4800000031a69e01 */
                                                                                               /* 0x2292928280429297 */
        /*0b48*/                   DMUL R12, R24, R24;                                         /* 0x5000000061831c01 */
        /*0b50*/                   DFMA R10, R26, R26, R12;                                    /* 0x2018000069a29c01 */
        /*0b58*/                   IADD32I R19, R11, -0x3500000;                               /* 0x0bf2c00000b4dc02 */
        /*0b60*/                   MUFU.RSQ64H R13, R11;                                       /* 0xc80000001cb35c00 */
        /*0b68*/                   MOV R12, R19;                                               /* 0x280000004c031de4 */
        /*0b70*/                   DMUL R14, R12, R12;                                         /* 0x5000000030c39c01 */
        /*0b78*/                   DFMA R14, -R10, R14, c[0x2][0x0];                           /* 0x201c880000a39e01 */
                                                                                               /* 0x2280420282029237 */
        /*0b88*/                   DFMA R16, R14, 0.375, R20;                                  /* 0x2028cff600e41c01 */
        /*0b90*/                   DMUL R14, R12, R14;                                         /* 0x5000000038c39c01 */
        /*0b98*/                   DFMA R20, R16, R14, R12;                                    /* 0x2018000039051c01 */
        /*0ba0*/                   ISETP.LT.U32.AND P0, PT, R19, c[0x2][0x8], PT;              /* 0x188e48002131dc03 */
        /*0ba8*/                   DMUL R16, R10, R20;                                         /* 0x5000000050a41c01 */
        /*0bb0*/                   IADD32I R23, R21, -0x100000;                                /* 0x0bffc0000155dc02 */
        /*0bb8*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
                                                                                               /* 0x22b20042b2028207 */
        /*0bc8*/                   DFMA R12, -R16, R16, R10;                                   /* 0x2014000041031e01 */
        /*0bd0*/                   MOV R15, R23;                                               /* 0x280000005c03dde4 */
        /*0bd8*/                   DFMA R28, R12, R14, R16;                                    /* 0x2020000038c71c01 */
        /*0be0*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*0be8*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*0bf0*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*0bf8*/                   MOV R19, R23;                                               /* 0x280000005c04dde4 */
                                                                                               /* 0x22028042c042b2b7 */
        /*0c08*/                   CAL 0x1bd0;                                                 /* 0x5000003f00011c07 */
        /*0c10*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0c18*/                   IMAD R22.CC, R37, 0x8, R18;                                 /* 0x2025c00022559ca3 */
        /*0c20*/                   LD.E.64 R12, [R4+0x28];                                     /* 0x84000000a0431ca5 */
        /*0c28*/                   IMAD.HI.X R23, R37, 0x8, R9;                                /* 0x2092c0002255dce3 */
        /*0c30*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*0c38*/                   LD.E.64 R10, [R22+0x18];                                    /* 0x8400000061629ca5 */
                                                                                               /* 0x2293f29292804297 */
        /*0c48*/                   DFMA R16, R28, R28, c[0x2][0x18];                           /* 0x2038880061c41c01 */
        /*0c50*/                   MUFU.RCP64H R15, R17;                                       /* 0xc80000001913dc00 */
        /*0c58*/                   SSY 0xd18;                                                  /* 0x60000002e0001c07 */
        /*0c60*/                   DFMA R18, -R16, R14, c[0x2][0x0];                           /* 0x201c880001049e01 */
        /*0c68*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*0c70*/                   DFMA R14, R14, R18, R14;                                    /* 0x201c000048e39c01 */
        /*0c78*/                   DMUL R10, R10, c[0x2][0x10];                                /* 0x5000480040a29c01 */
                                                                                               /* 0x22c2828202929297 */
        /*0c88*/                   DMUL R20, R10, R12;                                         /* 0x5000000030a51c01 */
        /*0c90*/                   DMUL R10, R20, R14;                                         /* 0x5000000039429c01 */
        /*0c98*/                   DFMA R12, -R16, R10, R20;                                   /* 0x2028000029031e01 */
        /*0ca0*/                   DFMA R30, R14, R12, R10;                                    /* 0x2014000030e79c01 */
        /*0ca8*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
        /*0cb0*/                   FFMA R9, RZ, R17, R31;                                      /* 0x303e000047f25c00 */
        /*0cb8*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
                                                                                               /* 0x2202b2b0420042b7 */
        /*0cc8*/               @P0 BRA CC.NEU, 0xd10;                                          /* 0x40000001000001a7 */
        /*0cd0*/                   MOV R10, R20;                                               /* 0x2800000050029de4 */
        /*0cd8*/                   MOV R11, R21;                                               /* 0x280000005402dde4 */
        /*0ce0*/                   MOV R9, R16;                                                /* 0x2800000040025de4 */
        /*0ce8*/                   MOV R12, R17;                                               /* 0x2800000044031de4 */
        /*0cf0*/                   CAL 0x1868;                                                 /* 0x5000002dc0011c07 */
        /*0cf8*/                   MOV R30, R16;                                               /* 0x2800000040079de4 */
                                                                                               /* 0x229292928042b2b7 */
        /*0d08*/                   MOV R31, R17;                                               /* 0x280000004407dde4 */
        /*0d10*/                   MUFU.RCP64H.S R11, R29;                                     /* 0xc800000019d2dc10 */
        /*0d18*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*0d20*/                   DMUL R14, R26, R30;                                         /* 0x5000000079a39c01 */
        /*0d28*/                   DFMA R12, -R28, R10, c[0x2][0x0];                           /* 0x2014880001c31e01 */
        /*0d30*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*0d38*/                   DFMA R10, R10, R12, R10;                                    /* 0x2014000030a29c01 */
                                                                                               /* 0x22c2828042029297 */
        /*0d48*/                   DMUL R16, R14, R10;                                         /* 0x5000000028e41c01 */
        /*0d50*/                   DFMA R12, -R28, R16, R14;                                   /* 0x201c000041c31e01 */
        /*0d58*/                   DFMA R16, R10, R12, R16;                                    /* 0x2020000030a41c01 */
        /*0d60*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0xc], PT;                 /* 0x160f480030ffdc80 */
        /*0d68*/                   SSY 0xdc0;                                                  /* 0x6000000140001c07 */
        /*0d70*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*0d78*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
                                                                                               /* 0x22b2b2b0420042b7 */
        /*0d88*/               @P0 BRA CC.NEU, 0xdb8;                                          /* 0x40000000a00001a7 */
        /*0d90*/                   MOV R10, R14;                                               /* 0x2800000038029de4 */
        /*0d98*/                   MOV R11, R15;                                               /* 0x280000003c02dde4 */
        /*0da0*/                   MOV R9, R28;                                                /* 0x2800000070025de4 */
        /*0da8*/                   MOV R12, R29;                                               /* 0x2800000074031de4 */
        /*0db0*/                   CAL 0x1868;                                                 /* 0x5000002ac0011c07 */
        /*0db8*/                   NOP.S;                                                      /* 0x4000000000001df4 */
                                                                                               /* 0x2292929280420047 */
        /*0dc8*/                   LD.E.64 R10, [R2+0x4e358];                                  /* 0x8400138d60229ca5 */
        /*0dd0*/                   MUFU.RCP64H R15, R29;                                       /* 0xc800000019d3dc00 */
        /*0dd8*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*0de0*/                   DMUL R24, R24, R30;                                         /* 0x5000000079861c01 */
        /*0de8*/                   DFMA R12, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c31e01 */
        /*0df0*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*0df8*/                   DFMA R18, R14, R12, R14;                                    /* 0x201c000030e49c01 */
                                                                                               /* 0x22829233f0420297 */
        /*0e08*/                   DMUL R14, R18, R24;                                         /* 0x5000000061239c01 */
        /*0e10*/                   DFMA R12, -R28, R14, R24;                                   /* 0x2030000039c31e01 */
        /*0e18*/                   FSET.GTU.AND RZ.CC, |R25|, c[0x2][0xc], PT;                 /* 0x160f4800319fdc80 */
        /*0e20*/                   SSY 0xea8;                                                  /* 0x6000000200001c07 */
        /*0e28*/                   DADD R16, R16, R10;                                         /* 0x4800000029041c01 */
        /*0e30*/                   DFMA R10, R18, R12, R14;                                    /* 0x201c000031229c01 */
        /*0e38*/                   FFMA R9, RZ, R29, R11;                                      /* 0x3016000077f25c00 */
                                                                                               /* 0x22b0420042b2c047 */
        /*0e48*/                   ST.E.64 [R2+0x4e358], R16;                                  /* 0x9400138d60241ca5 */
        /*0e50*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*0e58*/               @P0 BRA CC.NEU, 0xea0;                                          /* 0x40000001000001a7 */
        /*0e60*/                   MOV R10, R24;                                               /* 0x2800000060029de4 */
        /*0e68*/                   MOV R11, R25;                                               /* 0x280000006402dde4 */
        /*0e70*/                   MOV R9, R28;                                                /* 0x2800000070025de4 */
        /*0e78*/                   MOV R12, R29;                                               /* 0x2800000074031de4 */
                                                                                               /* 0x22b293f2b2b042b7 */
        /*0e88*/                   CAL 0x1868;                                                 /* 0x5000002760011c07 */
        /*0e90*/                   MOV R10, R16;                                               /* 0x2800000040029de4 */
        /*0e98*/                   MOV R11, R17;                                               /* 0x280000004402dde4 */
        /*0ea0*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0ea8*/                   LD.E.64 R12, [R2+0x5dd98];                                  /* 0x8400177660231ca5 */
        /*0eb0*/                   DADD R12, R10, R12;                                         /* 0x4800000030a31c01 */
        /*0eb8*/                   ST.E.64.S [R2+0x5dd98], R12;                                /* 0x9400177660231cb5 */
                                                                                               /* 0x22028042b2c2c047 */
        /*0ec8*/                   ISUB RZ.CC, R7, R33;                                        /* 0x48010000847fdd03 */
        /*0ed0*/                   SSY 0x1340;                                                 /* 0x60000011a0001c07 */
        /*0ed8*/                   ISETP.EQ.X.AND P0, PT, R39, R36, PT;                        /* 0x190e00009271dc63 */
        /*0ee0*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*0ee8*/                   MOV32I R9, 0x3;                                             /* 0x180000000c025de2 */
        /*0ef0*/                   SHL R10, R7, -0x1d;                                         /* 0x6000ffff8c729c03 */
        /*0ef8*/                   IADD32I RZ.CC, R9, -0x20;                                   /* 0x0fffffff809fdc02 */
                                                                                               /* 0x220042c23042b047 */
        /*0f08*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
        /*0f10*/                   SHL R11, R39, 0x3;                                          /* 0x6000c0000e72dc03 */
        /*0f18*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*0f20*/                   SSY 0x1098;                                                 /* 0x60000005c0001c07 */
        /*0f28*/                   ISCADD R9.CC, R37, 0x8, 0x3;                                /* 0x4001c00022525c63 */
        /*0f30*/                   IMAD.HI.X R13, R37, 0x8, RZ;                                /* 0x20fec00022535ce3 */
        /*0f38*/                   IADD R20.CC, R9, R18;                                       /* 0x4801000048951c03 */
                                                                                               /* 0x228042c04202b047 */
        /*0f48*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*0f50*/                   SHR.U32 R12, R7, 0x1d;                                      /* 0x5800c00074731c03 */
        /*0f58*/                   IADD.X R21, R13, R9;                                        /* 0x4800000024d55c43 */
        /*0f60*/                   IMAD.U32.U32 R16.CC, R7, 0x8, R18;                          /* 0x2025c00020741c03 */
        /*0f68*/               @P0 IADD R10, R11, R12;                                         /* 0x4800000030b28003 */
        /*0f70*/                   IADD.X R17, R10, R9;                                        /* 0x4800000024a45c43 */
        /*0f78*/                   LD.E.64 R24, [R20+0x1f490];                                 /* 0x840007d241461ca5 */
                                                                                               /* 0x223273f042a23237 */
        /*0f88*/                   LD.E.64 R10, [R16+0x1f490];                                 /* 0x840007d241029ca5 */
        /*0f90*/                   LD.E.64 R26, [R20+0xfa50];                                  /* 0x840003e941469ca5 */
        /*0f98*/                   LD.E.64 R12, [R16+0xfa50];                                  /* 0x840003e941031ca5 */
        /*0fa0*/                   MOV32I R20, 0x0;                                            /* 0x1800000000051de2 */
        /*0fa8*/                   MOV32I R21, 0x3fe00000;                                     /* 0x18ff800000055de2 */
        /*0fb0*/                   DADD R24, -R24, R10;                                        /* 0x4800000029861e01 */
        /*0fb8*/                   DADD R26, -R26, R12;                                        /* 0x4800000031a69e01 */
                                                                                               /* 0x2292928280429297 */
        /*0fc8*/                   DMUL R12, R24, R24;                                         /* 0x5000000061831c01 */
        /*0fd0*/                   DFMA R10, R26, R26, R12;                                    /* 0x2018000069a29c01 */
        /*0fd8*/                   IADD32I R19, R11, -0x3500000;                               /* 0x0bf2c00000b4dc02 */
        /*0fe0*/                   MUFU.RSQ64H R13, R11;                                       /* 0xc80000001cb35c00 */
        /*0fe8*/                   MOV R12, R19;                                               /* 0x280000004c031de4 */
        /*0ff0*/                   DMUL R14, R12, R12;                                         /* 0x5000000030c39c01 */
        /*0ff8*/                   DFMA R14, -R10, R14, c[0x2][0x0];                           /* 0x201c880000a39e01 */
                                                                                               /* 0x2280420282029237 */
        /*1008*/                   DFMA R16, R14, 0.375, R20;                                  /* 0x2028cff600e41c01 */
        /*1010*/                   DMUL R14, R12, R14;                                         /* 0x5000000038c39c01 */
        /*1018*/                   DFMA R20, R16, R14, R12;                                    /* 0x2018000039051c01 */
        /*1020*/                   ISETP.LT.U32.AND P0, PT, R19, c[0x2][0x8], PT;              /* 0x188e48002131dc03 */
        /*1028*/                   DMUL R16, R10, R20;                                         /* 0x5000000050a41c01 */
        /*1030*/                   IADD32I R23, R21, -0x100000;                                /* 0x0bffc0000155dc02 */
        /*1038*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
                                                                                               /* 0x22b20042b2028207 */
        /*1048*/                   DFMA R12, -R16, R16, R10;                                   /* 0x2014000041031e01 */
        /*1050*/                   MOV R15, R23;                                               /* 0x280000005c03dde4 */
        /*1058*/                   DFMA R28, R12, R14, R16;                                    /* 0x2020000038c71c01 */
        /*1060*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*1068*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*1070*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*1078*/                   MOV R19, R23;                                               /* 0x280000005c04dde4 */
                                                                                               /* 0x22028042c042b2b7 */
        /*1088*/                   CAL 0x1bd0;                                                 /* 0x5000002d00011c07 */
        /*1090*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*1098*/                   IMAD R22.CC, R37, 0x8, R18;                                 /* 0x2025c00022559ca3 */
        /*10a0*/                   LD.E.64 R12, [R4+0x30];                                     /* 0x84000000c0431ca5 */
        /*10a8*/                   IMAD.HI.X R23, R37, 0x8, R9;                                /* 0x2092c0002255dce3 */
        /*10b0*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*10b8*/                   LD.E.64 R10, [R22+0x18];                                    /* 0x8400000061629ca5 */
                                                                                               /* 0x2293f29292804297 */
        /*10c8*/                   DFMA R16, R28, R28, c[0x2][0x18];                           /* 0x2038880061c41c01 */
        /*10d0*/                   MUFU.RCP64H R15, R17;                                       /* 0xc80000001913dc00 */
        /*10d8*/                   SSY 0x1198;                                                 /* 0x60000002e0001c07 */
        /*10e0*/                   DFMA R18, -R16, R14, c[0x2][0x0];                           /* 0x201c880001049e01 */
        /*10e8*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*10f0*/                   DFMA R14, R14, R18, R14;                                    /* 0x201c000048e39c01 */
        /*10f8*/                   DMUL R10, R10, c[0x2][0x10];                                /* 0x5000480040a29c01 */
                                                                                               /* 0x22c2828202929297 */
        /*1108*/                   DMUL R20, R10, R12;                                         /* 0x5000000030a51c01 */
        /*1110*/                   DMUL R10, R20, R14;                                         /* 0x5000000039429c01 */
        /*1118*/                   DFMA R12, -R16, R10, R20;                                   /* 0x2028000029031e01 */
        /*1120*/                   DFMA R30, R14, R12, R10;                                    /* 0x2014000030e79c01 */
        /*1128*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
        /*1130*/                   FFMA R9, RZ, R17, R31;                                      /* 0x303e000047f25c00 */
        /*1138*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
                                                                                               /* 0x2202b2b0420042b7 */
        /*1148*/               @P0 BRA CC.NEU, 0x1190;                                         /* 0x40000001000001a7 */
        /*1150*/                   MOV R10, R20;                                               /* 0x2800000050029de4 */
        /*1158*/                   MOV R11, R21;                                               /* 0x280000005402dde4 */
        /*1160*/                   MOV R9, R16;                                                /* 0x2800000040025de4 */
        /*1168*/                   MOV R12, R17;                                               /* 0x2800000044031de4 */
        /*1170*/                   CAL 0x1868;                                                 /* 0x5000001bc0011c07 */
        /*1178*/                   MOV R30, R16;                                               /* 0x2800000040079de4 */
                                                                                               /* 0x229292928042b2b7 */
        /*1188*/                   MOV R31, R17;                                               /* 0x280000004407dde4 */
        /*1190*/                   MUFU.RCP64H.S R11, R29;                                     /* 0xc800000019d2dc10 */
        /*1198*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*11a0*/                   DMUL R14, R26, R30;                                         /* 0x5000000079a39c01 */
        /*11a8*/                   DFMA R12, -R28, R10, c[0x2][0x0];                           /* 0x2014880001c31e01 */
        /*11b0*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*11b8*/                   DFMA R10, R10, R12, R10;                                    /* 0x2014000030a29c01 */
                                                                                               /* 0x22c2828042029297 */
        /*11c8*/                   DMUL R16, R14, R10;                                         /* 0x5000000028e41c01 */
        /*11d0*/                   DFMA R12, -R28, R16, R14;                                   /* 0x201c000041c31e01 */
        /*11d8*/                   DFMA R16, R10, R12, R16;                                    /* 0x2020000030a41c01 */
        /*11e0*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0xc], PT;                 /* 0x160f480030ffdc80 */
        /*11e8*/                   SSY 0x1240;                                                 /* 0x6000000140001c07 */
        /*11f0*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*11f8*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
                                                                                               /* 0x22b2b2b0420042b7 */
        /*1208*/               @P0 BRA CC.NEU, 0x1238;                                         /* 0x40000000a00001a7 */
        /*1210*/                   MOV R10, R14;                                               /* 0x2800000038029de4 */
        /*1218*/                   MOV R11, R15;                                               /* 0x280000003c02dde4 */
        /*1220*/                   MOV R9, R28;                                                /* 0x2800000070025de4 */
        /*1228*/                   MOV R12, R29;                                               /* 0x2800000074031de4 */
        /*1230*/                   CAL 0x1868;                                                 /* 0x50000018c0011c07 */
        /*1238*/                   NOP.S;                                                      /* 0x4000000000001df4 */
                                                                                               /* 0x2292929280420047 */
        /*1248*/                   LD.E.64 R10, [R2+0x4e358];                                  /* 0x8400138d60229ca5 */
        /*1250*/                   MUFU.RCP64H R15, R29;                                       /* 0xc800000019d3dc00 */
        /*1258*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*1260*/                   DMUL R24, R24, R30;                                         /* 0x5000000079861c01 */
        /*1268*/                   DFMA R12, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c31e01 */
        /*1270*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*1278*/                   DFMA R18, R14, R12, R14;                                    /* 0x201c000030e49c01 */
                                                                                               /* 0x22829233f0420297 */
        /*1288*/                   DMUL R14, R18, R24;                                         /* 0x5000000061239c01 */
        /*1290*/                   DFMA R12, -R28, R14, R24;                                   /* 0x2030000039c31e01 */
        /*1298*/                   FSET.GTU.AND RZ.CC, |R25|, c[0x2][0xc], PT;                 /* 0x160f4800319fdc80 */
        /*12a0*/                   SSY 0x1328;                                                 /* 0x6000000200001c07 */
        /*12a8*/                   DADD R16, R16, R10;                                         /* 0x4800000029041c01 */
        /*12b0*/                   DFMA R10, R18, R12, R14;                                    /* 0x201c000031229c01 */
        /*12b8*/                   FFMA R9, RZ, R29, R11;                                      /* 0x3016000077f25c00 */
                                                                                               /* 0x22b0420042b2c047 */
        /*12c8*/                   ST.E.64 [R2+0x4e358], R16;                                  /* 0x9400138d60241ca5 */
        /*12d0*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*12d8*/               @P0 BRA CC.NEU, 0x1320;                                         /* 0x40000001000001a7 */
        /*12e0*/                   MOV R10, R24;                                               /* 0x2800000060029de4 */
        /*12e8*/                   MOV R11, R25;                                               /* 0x280000006402dde4 */
        /*12f0*/                   MOV R9, R28;                                                /* 0x2800000070025de4 */
        /*12f8*/                   MOV R12, R29;                                               /* 0x2800000074031de4 */
                                                                                               /* 0x22b293f2b2b042b7 */
        /*1308*/                   CAL 0x1868;                                                 /* 0x5000001560011c07 */
        /*1310*/                   MOV R10, R16;                                               /* 0x2800000040029de4 */
        /*1318*/                   MOV R11, R17;                                               /* 0x280000004402dde4 */
        /*1320*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*1328*/                   LD.E.64 R12, [R2+0x5dd98];                                  /* 0x8400177660231ca5 */
        /*1330*/                   DADD R12, R10, R12;                                         /* 0x4800000030a31c01 */
        /*1338*/                   ST.E.64.S [R2+0x5dd98], R12;                                /* 0x9400177660231cb5 */
                                                                                               /* 0x22028042b2c2c047 */
        /*1348*/                   ISUB RZ.CC, R8, R33;                                        /* 0x48010000848fdd03 */
        /*1350*/                   SSY 0x17c0;                                                 /* 0x60000011a0001c07 */
        /*1358*/                   ISETP.EQ.X.AND P0, PT, R38, R36, PT;                        /* 0x190e00009261dc63 */
        /*1360*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*1368*/                   MOV32I R9, 0x3;                                             /* 0x180000000c025de2 */
        /*1370*/                   SHL R10, R8, -0x1d;                                         /* 0x6000ffff8c829c03 */
        /*1378*/                   IADD32I RZ.CC, R9, -0x20;                                   /* 0x0fffffff809fdc02 */
                                                                                               /* 0x220042c23042b047 */
        /*1388*/                   MOV32I R18, 0x0;                                            /* 0x1800000000049de2 */
        /*1390*/                   SHL R11, R38, 0x3;                                          /* 0x6000c0000e62dc03 */
        /*1398*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*13a0*/                   SSY 0x1518;                                                 /* 0x60000005c0001c07 */
        /*13a8*/                   ISCADD R9.CC, R37, 0x8, 0x3;                                /* 0x4001c00022525c63 */
        /*13b0*/                   IMAD.HI.X R13, R37, 0x8, RZ;                                /* 0x20fec00022535ce3 */
        /*13b8*/                   IADD R20.CC, R9, R18;                                       /* 0x4801000048951c03 */
                                                                                               /* 0x228280428042b047 */
        /*13c8*/                   MOV32I R9, 0x0;                                             /* 0x1800000000025de2 */
        /*13d0*/                   SHR.U32 R12, R8, 0x1d;                                      /* 0x5800c00074831c03 */
        /*13d8*/                   IADD.X R21, R13, R9;                                        /* 0x4800000024d55c43 */
        /*13e0*/                   IMAD.U32.U32 R16.CC, R8, 0x8, R18;                          /* 0x2025c00020841c03 */
        /*13e8*/               @P0 IADD R10, R11, R12;                                         /* 0x4800000030b28003 */
        /*13f0*/                   LD.E.64 R24, [R20+0x1f490];                                 /* 0x840007d241461ca5 */
        /*13f8*/                   IADD.X R17, R10, R9;                                        /* 0x4800000024a45c43 */
                                                                                               /* 0x223273f042a23237 */
        /*1408*/                   LD.E.64 R10, [R16+0x1f490];                                 /* 0x840007d241029ca5 */
        /*1410*/                   LD.E.64 R26, [R20+0xfa50];                                  /* 0x840003e941469ca5 */
        /*1418*/                   LD.E.64 R12, [R16+0xfa50];                                  /* 0x840003e941031ca5 */
        /*1420*/                   MOV32I R20, 0x0;                                            /* 0x1800000000051de2 */
        /*1428*/                   MOV32I R21, 0x3fe00000;                                     /* 0x18ff800000055de2 */
        /*1430*/                   DADD R24, -R24, R10;                                        /* 0x4800000029861e01 */
        /*1438*/                   DADD R26, -R26, R12;                                        /* 0x4800000031a69e01 */
                                                                                               /* 0x2292928280429297 */
        /*1448*/                   DMUL R12, R24, R24;                                         /* 0x5000000061831c01 */
        /*1450*/                   DFMA R10, R26, R26, R12;                                    /* 0x2018000069a29c01 */
        /*1458*/                   IADD32I R19, R11, -0x3500000;                               /* 0x0bf2c00000b4dc02 */
        /*1460*/                   MUFU.RSQ64H R13, R11;                                       /* 0xc80000001cb35c00 */
        /*1468*/                   MOV R12, R19;                                               /* 0x280000004c031de4 */
        /*1470*/                   DMUL R14, R12, R12;                                         /* 0x5000000030c39c01 */
        /*1478*/                   DFMA R14, -R10, R14, c[0x2][0x0];                           /* 0x201c880000a39e01 */
                                                                                               /* 0x2280420282029237 */
        /*1488*/                   DFMA R16, R14, 0.375, R20;                                  /* 0x2028cff600e41c01 */
        /*1490*/                   DMUL R14, R12, R14;                                         /* 0x5000000038c39c01 */
        /*1498*/                   DFMA R20, R16, R14, R12;                                    /* 0x2018000039051c01 */
        /*14a0*/                   ISETP.LT.U32.AND P0, PT, R19, c[0x2][0x8], PT;              /* 0x188e48002131dc03 */
        /*14a8*/                   DMUL R16, R10, R20;                                         /* 0x5000000050a41c01 */
        /*14b0*/                   IADD32I R23, R21, -0x100000;                                /* 0x0bffc0000155dc02 */
        /*14b8*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
                                                                                               /* 0x22b20042b2028207 */
        /*14c8*/                   DFMA R12, -R16, R16, R10;                                   /* 0x2014000041031e01 */
        /*14d0*/                   MOV R15, R23;                                               /* 0x280000005c03dde4 */
        /*14d8*/                   DFMA R28, R12, R14, R16;                                    /* 0x2020000038c71c01 */
        /*14e0*/               @P0 NOP.S;                                                      /* 0x40000000000001f4 */
        /*14e8*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*14f0*/                   MOV R14, R20;                                               /* 0x2800000050039de4 */
        /*14f8*/                   MOV R19, R23;                                               /* 0x280000005c04dde4 */
                                                                                               /* 0x22028042c042b2b7 */
        /*1508*/                   CAL 0x1bd0;                                                 /* 0x5000001b00011c07 */
        /*1510*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*1518*/                   IMAD R22.CC, R37, 0x8, R18;                                 /* 0x2025c00022559ca3 */
        /*1520*/                   LD.E.64 R12, [R4+0x38];                                     /* 0x84000000e0431ca5 */
        /*1528*/                   IMAD.HI.X R23, R37, 0x8, R9;                                /* 0x2092c0002255dce3 */
        /*1530*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*1538*/                   LD.E.64 R10, [R22+0x18];                                    /* 0x8400000061629ca5 */
                                                                                               /* 0x2293f29292804297 */
        /*1548*/                   DFMA R16, R28, R28, c[0x2][0x18];                           /* 0x2038880061c41c01 */
        /*1550*/                   MUFU.RCP64H R15, R17;                                       /* 0xc80000001913dc00 */
        /*1558*/                   SSY 0x1618;                                                 /* 0x60000002e0001c07 */
        /*1560*/                   DFMA R18, -R16, R14, c[0x2][0x0];                           /* 0x201c880001049e01 */
        /*1568*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*1570*/                   DFMA R14, R14, R18, R14;                                    /* 0x201c000048e39c01 */
        /*1578*/                   DMUL R10, R10, c[0x2][0x10];                                /* 0x5000480040a29c01 */
                                                                                               /* 0x22c2828202929297 */
        /*1588*/                   DMUL R20, R10, R12;                                         /* 0x5000000030a51c01 */
        /*1590*/                   DMUL R10, R20, R14;                                         /* 0x5000000039429c01 */
        /*1598*/                   DFMA R12, -R16, R10, R20;                                   /* 0x2028000029031e01 */
        /*15a0*/                   DFMA R30, R14, R12, R10;                                    /* 0x2014000030e79c01 */
        /*15a8*/                   FSET.GTU.AND RZ.CC, |R21|, c[0x2][0xc], PT;                 /* 0x160f4800315fdc80 */
        /*15b0*/                   FFMA R9, RZ, R17, R31;                                      /* 0x303e000047f25c00 */
        /*15b8*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
                                                                                               /* 0x2202b2b0420042b7 */
        /*15c8*/               @P0 BRA CC.NEU, 0x1610;                                         /* 0x40000001000001a7 */
        /*15d0*/                   MOV R10, R20;                                               /* 0x2800000050029de4 */
        /*15d8*/                   MOV R11, R21;                                               /* 0x280000005402dde4 */
        /*15e0*/                   MOV R9, R16;                                                /* 0x2800000040025de4 */
        /*15e8*/                   MOV R12, R17;                                               /* 0x2800000044031de4 */
        /*15f0*/                   CAL 0x1868;                                                 /* 0x50000009c0011c07 */
        /*15f8*/                   MOV R30, R16;                                               /* 0x2800000040079de4 */
                                                                                               /* 0x229292928042b2b7 */
        /*1608*/                   MOV R31, R17;                                               /* 0x280000004407dde4 */
        /*1610*/                   MUFU.RCP64H.S R11, R29;                                     /* 0xc800000019d2dc10 */
        /*1618*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*1620*/                   DMUL R14, R26, R30;                                         /* 0x5000000079a39c01 */
        /*1628*/                   DFMA R12, -R28, R10, c[0x2][0x0];                           /* 0x2014880001c31e01 */
        /*1630*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*1638*/                   DFMA R10, R10, R12, R10;                                    /* 0x2014000030a29c01 */
                                                                                               /* 0x22c2828042029297 */
        /*1648*/                   DMUL R16, R14, R10;                                         /* 0x5000000028e41c01 */
        /*1650*/                   DFMA R12, -R28, R16, R14;                                   /* 0x201c000041c31e01 */
        /*1658*/                   DFMA R16, R10, R12, R16;                                    /* 0x2020000030a41c01 */
        /*1660*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0xc], PT;                 /* 0x160f480030ffdc80 */
        /*1668*/                   SSY 0x16c0;                                                 /* 0x6000000140001c07 */
        /*1670*/                   FFMA R9, RZ, R29, R17;                                      /* 0x3022000077f25c00 */
        /*1678*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
                                                                                               /* 0x22b2b2b0420042b7 */
        /*1688*/               @P0 BRA CC.NEU, 0x16b8;                                         /* 0x40000000a00001a7 */
        /*1690*/                   MOV R10, R14;                                               /* 0x2800000038029de4 */
        /*1698*/                   MOV R11, R15;                                               /* 0x280000003c02dde4 */
        /*16a0*/                   MOV R9, R28;                                                /* 0x2800000070025de4 */
        /*16a8*/                   MOV R12, R29;                                               /* 0x2800000074031de4 */
        /*16b0*/                   CAL 0x1868;                                                 /* 0x50000006c0011c07 */
        /*16b8*/                   NOP.S;                                                      /* 0x4000000000001df4 */
                                                                                               /* 0x2292929280420047 */
        /*16c8*/                   LD.E.64 R10, [R2+0x4e358];                                  /* 0x8400138d60229ca5 */
        /*16d0*/                   MUFU.RCP64H R15, R29;                                       /* 0xc800000019d3dc00 */
        /*16d8*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*16e0*/                   DMUL R24, R24, R30;                                         /* 0x5000000079861c01 */
        /*16e8*/                   DFMA R12, -R28, R14, c[0x2][0x0];                           /* 0x201c880001c31e01 */
        /*16f0*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*16f8*/                   DFMA R18, R14, R12, R14;                                    /* 0x201c000030e49c01 */
                                                                                               /* 0x22829233f0420297 */
        /*1708*/                   DMUL R14, R18, R24;                                         /* 0x5000000061239c01 */
        /*1710*/                   DFMA R12, -R28, R14, R24;                                   /* 0x2030000039c31e01 */
        /*1718*/                   FSET.GTU.AND RZ.CC, |R25|, c[0x2][0xc], PT;                 /* 0x160f4800319fdc80 */
        /*1720*/                   SSY 0x17a8;                                                 /* 0x6000000200001c07 */
        /*1728*/                   DADD R16, R16, R10;                                         /* 0x4800000029041c01 */
        /*1730*/                   DFMA R10, R18, R12, R14;                                    /* 0x201c000031229c01 */
        /*1738*/                   FFMA R9, RZ, R29, R11;                                      /* 0x3016000077f25c00 */
                                                                                               /* 0x22b0420042b2c047 */
        /*1748*/                   ST.E.64 [R2+0x4e358], R16;                                  /* 0x9400138d60241ca5 */
        /*1750*/                   FSETP.GT.AND P0, PT, |R9|, c[0x2][0x20], PT;                /* 0x220e48008091dc80 */
        /*1758*/               @P0 BRA CC.NEU, 0x17a0;                                         /* 0x40000001000001a7 */
        /*1760*/                   MOV R10, R24;                                               /* 0x2800000060029de4 */
        /*1768*/                   MOV R11, R25;                                               /* 0x280000006402dde4 */
        /*1770*/                   MOV R9, R28;                                                /* 0x2800000070025de4 */
        /*1778*/                   MOV R12, R29;                                               /* 0x2800000074031de4 */
                                                                                               /* 0x22b293f2b2b042b7 */
        /*1788*/                   CAL 0x1868;                                                 /* 0x5000000360011c07 */
        /*1790*/                   MOV R10, R16;                                               /* 0x2800000040029de4 */
        /*1798*/                   MOV R11, R17;                                               /* 0x280000004402dde4 */
        /*17a0*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*17a8*/                   LD.E.64 R12, [R2+0x5dd98];                                  /* 0x8400177660231ca5 */
        /*17b0*/                   DADD R12, R10, R12;                                         /* 0x4800000030a31c01 */
        /*17b8*/                   ST.E.64.S [R2+0x5dd98], R12;                                /* 0x9400177660231cb5 */
                                                                                               /* 0x22c042c202c042c7 */
        /*17c8*/                   IADD32I R44.CC, R44, -0x5;                                  /* 0x0fffffffeecb1c02 */
        /*17d0*/                   IADD32I.X R43, R43, -0x1;                                   /* 0x0bfffffffebadc42 */
        /*17d8*/                   IADD RZ.CC, -RZ, R44;                                       /* 0x48010000b3ffde03 */
        /*17e0*/                   ISETP.NE.X.AND P0, PT, R43, RZ, PT;                         /* 0x1a8e0000feb1dc63 */
        /*17e8*/                   IADD32I R4.CC, R4, 0x28;                                    /* 0x0c000000a0411c02 */
        /*17f0*/                   IADD.X R5, RZ, R5;                                          /* 0x4800000017f15c43 */
        /*17f8*/                   IADD32I R8.CC, R8, 0x5;                                     /* 0x0c00000014821c02 */
                                                                                               /* 0x2202c202c202c047 */
        /*1808*/                   IADD.X R38, RZ, R38;                                        /* 0x480000009bf99c43 */
        /*1810*/                   IADD32I R7.CC, R7, 0x5;                                     /* 0x0c0000001471dc02 */
        /*1818*/                   IADD.X R39, RZ, R39;                                        /* 0x480000009ff9dc43 */
        /*1820*/                   IADD32I R6.CC, R6, 0x5;                                     /* 0x0c00000014619c02 */
        /*1828*/                   IADD.X R40, RZ, R40;                                        /* 0x48000000a3fa1c43 */
        /*1830*/                   IADD32I R0.CC, R0, 0x5;                                     /* 0x0c00000014001c02 */
        /*1838*/                   IADD.X R41, RZ, R41;                                        /* 0x48000000a7fa5c43 */
                                                                                               /* 0x22028042e2e042c7 */
        /*1848*/                   IADD32I R32.CC, R32, 0x5;                                   /* 0x0c00000016081c02 */
        /*1850*/                   IADD.X R42, RZ, R42;                                        /* 0x48000000abfa9c43 */
        /*1858*/               @P0 BRA 0x140;                                                  /* 0x4003ffa3800001e7 */
        /*1860*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*1868*/                   LOP32I.AND R13, R12, 0x40000000;                            /* 0x3900000000c35c02 */
        /*1870*/                   MOV32I R45, 0x5ff00000;                                     /* 0x197fc000000b5de2 */
        /*1878*/                   ISETP.LT.U32.AND P0, PT, R13, c[0x2][0x24], PT;             /* 0x188e480090d1dc03 */
                                                                                               /* 0x228200428042b047 */
        /*1888*/                   MOV R13, R12;                                               /* 0x2800000030035de4 */
        /*1890*/                   MOV R12, R9;                                                /* 0x2800000024031de4 */
        /*1898*/                   SEL R14, R45, c[0x2][0x28], P0;                             /* 0x20004800a2d39c04 */
        /*18a0*/                   LOP32I.AND R9, R11, 0x7f800000;                             /* 0x39fe000000b25c02 */
        /*18a8*/                   MOV R15, R14;                                               /* 0x280000003803dde4 */
        /*18b0*/                   PBK 0x1bc0;                                                 /* 0x6800000c20001c07 */
        /*18b8*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
                                                                                               /* 0x228200428042c207 */
        /*18c8*/                   DMUL R16, R12, R14;                                         /* 0x5000000038c41c01 */
        /*18d0*/                   ISETP.LT.U32.AND P0, PT, R9, c[0x2][0x2c], PT;              /* 0x188e4800b091dc03 */
        /*18d8*/                   MUFU.RCP64H R20, R17;                                       /* 0xc800000019151c00 */
        /*18e0*/                   SEL R34, R45, c[0x2][0x28], P0;                             /* 0x20004800a2d89c04 */
        /*18e8*/                   MOV R21, R20;                                               /* 0x2800000050055de4 */
        /*18f0*/                   MOV R35, R34;                                               /* 0x280000008808dde4 */
        /*18f8*/                   MOV32I R20, 0x1;                                            /* 0x1800000004051de2 */
                                                                                               /* 0x2292929252328207 */
        /*1908*/                   DFMA R18, -R16, R20, c[0x2][0x0];                           /* 0x2028880001049e01 */
        /*1910*/                   MOV R34, RZ;                                                /* 0x28000000fc089de4 */
        /*1918*/                   DFMA R22, R18, R18, R18;                                    /* 0x2024000049259c01 */
        /*1920*/                   DMUL R18, R10, R34;                                         /* 0x5000000088a49c01 */
        /*1928*/                   DFMA R20, R20, R22, R20;                                    /* 0x2028000059451c01 */
        /*1930*/                   DMUL R22, R18, R20;                                         /* 0x5000000051259c01 */
        /*1938*/                   DFMA R16, -R16, R22, R18;                                   /* 0x2024000059041e01 */
                                                                                               /* 0x220282c042b2d297 */
        /*1948*/                   DFMA R20, R20, R16, R22;                                    /* 0x202c000041451c01 */
        /*1950*/                   DSETP.LEU.AND P0, PT, |R20|, RZ, PT;                        /* 0x1d8e0000fd41dc81 */
        /*1958*/               @P0 BRA 0x1b38;                                                 /* 0x40000007600001e7 */
        /*1960*/                   ISETP.GT.U32.AND P0, PT, R9, c[0x2][0x30], PT;              /* 0x1a0e4800c091dc03 */
        /*1968*/                   DMUL R18, R14, R20;                                         /* 0x5000000050e49c01 */
        /*1970*/                   SEL R16, R45, c[0x2][0x28], P0;                             /* 0x20004800a2d41c04 */
        /*1978*/                   MOV R17, R16;                                               /* 0x2800000040045de4 */
                                                                                               /* 0x22d2925232523287 */
        /*1988*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*1990*/                   DMUL R20, R20, R16;                                         /* 0x5000000041451c01 */
        /*1998*/                   DMUL R18, R16, R18;                                         /* 0x5000000049049c01 */
        /*19a0*/                   DMUL R20, R14, R20;                                         /* 0x5000000050e51c01 */
        /*19a8*/                   DFMA R14, R12, R18, -R10;                                   /* 0x2014000048c39d01 */
        /*19b0*/                   DFMA R16, R12, R20, -R10;                                   /* 0x2014000050c41d01 */
        /*19b8*/                   DSETP.GT.AND P0, PT, |R14|, |R16|, PT;                      /* 0x1a0e000040e1dcc1 */
                                                                                               /* 0x220042b042c27207 */
        /*19c8*/                   SEL R9, R21, R19, P0;                                       /* 0x200000004d525c04 */
        /*19d0*/                   SEL R16, R20, R18, P0;                                      /* 0x2000000049441c04 */
        /*19d8*/                   FSETP.GTU.AND P0, PT, |R9|, 1.469367938527859385e-39, PT;   /* 0x260ec0040091dc80 */
        /*19e0*/                   MOV R17, R9;                                                /* 0x2800000024045de4 */
        /*19e8*/               @P0 BRK;                                                        /* 0xa8000000000001e7 */
        /*19f0*/                   FSETP.LT.AND P0, PT, |R11|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000b1dc80 */
        /*19f8*/                   LOP32I.AND R19, R16, 0xfffffffe;                            /* 0x3bfffffff904dc02 */
                                                                                               /* 0x2280428042a20047 */
        /*1a08*/                   MOV32I R14, 0x58500000;                                     /* 0x1961400000039de2 */
        /*1a10*/                   MOV R17, R9;                                                /* 0x2800000024045de4 */
        /*1a18*/                   LOP32I.OR R18, R16, 0x1;                                    /* 0x3800000005049c42 */
        /*1a20*/                   SEL R14, R14, c[0x2][0x4], P0;                              /* 0x2000480010e39c04 */
        /*1a28*/                   MOV R16, R18;                                               /* 0x2800000048041de4 */
        /*1a30*/                   MOV R15, R14;                                               /* 0x280000003803dde4 */
        /*1a38*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
                                                                                               /* 0x22d2923272020237 */
        /*1a48*/                   DMUL R12, R12, R14;                                         /* 0x5000000038c31c01 */
        /*1a50*/                   DMUL R10, R10, R14;                                         /* 0x5000000038a29c01 */
        /*1a58*/                   MOV R14, R19;                                               /* 0x280000004c039de4 */
        /*1a60*/                   MOV R15, R9;                                                /* 0x280000002403dde4 */
        /*1a68*/                   DFMA R16, R16, R12, -R10;                                   /* 0x2014000031041d01 */
        /*1a70*/                   DFMA R14, R14, R12, -R10;                                   /* 0x2014000030e39d01 */
        /*1a78*/                   DSETP.GT.AND P0, PT, |R14|, |R16|, PT;                      /* 0x1a0e000040e1dcc1 */
                                                                                               /* 0x2202b20042c04287 */
        /*1a88*/                   SEL R14, R18, R19, P0;                                      /* 0x200000004d239c04 */
        /*1a90*/                   IADD32I R19.CC, R14, 0x1;                                   /* 0x0c00000004e4dc02 */
        /*1a98*/                   LOP32I.AND R15, R14, 0x1;                                   /* 0x3800000004e3dc02 */
        /*1aa0*/                   IADD.X R18, RZ, R9;                                         /* 0x4800000027f49c43 */
        /*1aa8*/                   IADD32I R16.CC, R14, -0x1;                                  /* 0x0ffffffffce41c02 */
        /*1ab0*/                   ISETP.EQ.U32.AND P0, PT, R15, 0x1, PT;                      /* 0x190ec00004f1dc03 */
        /*1ab8*/                   IADD32I.X R15, R9, -0x1;                                    /* 0x0bfffffffc93dc42 */
                                                                                               /* 0x2222520202026207 */
        /*1ac8*/                   SEL R19, R14, R19, !P0;                                     /* 0x201000004ce4dc04 */
        /*1ad0*/                   SEL R18, R9, R18, !P0;                                      /* 0x2010000048949c04 */
        /*1ad8*/                   SEL R9, R15, R9, !P0;                                       /* 0x2010000024f25c04 */
        /*1ae0*/                   SEL R14, R16, R14, !P0;                                     /* 0x2010000039039c04 */
        /*1ae8*/                   MOV R16, R19;                                               /* 0x280000004c041de4 */
        /*1af0*/                   MOV R17, R18;                                               /* 0x2800000048045de4 */
        /*1af8*/                   MOV R15, R9;                                                /* 0x280000002403dde4 */
                                                                                               /* 0x22d2e04202d29257 */
        /*1b08*/                   DFMA R16, R12, R16, -R10;                                   /* 0x2014000040c41d01 */
        /*1b10*/                   DFMA R10, R12, R14, -R10;                                   /* 0x2014000038c29d01 */
        /*1b18*/                   DSETP.GT.AND P0, PT, |R16|, |R10|, PT;                      /* 0x1a0e00002901dcc1 */
        /*1b20*/                   SEL R16, R14, R19, P0;                                      /* 0x200000004ce41c04 */
        /*1b28*/                   SEL R17, R9, R18, P0;                                       /* 0x2000000048945c04 */
        /*1b30*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1b38*/                   DSETP.EQ.AND P0, PT, R20, RZ, PT;                           /* 0x190e0000fd41dc01 */
                                                                                               /* 0x2202d2d2804282b7 */
        /*1b48*/               @P0 BRA 0x1bb0;                                                 /* 0x40000001800001e7 */
        /*1b50*/                   MUFU.RCP64H R14, R13;                                       /* 0xc800000018d39c00 */
        /*1b58*/                   MOV R15, R14;                                               /* 0x280000003803dde4 */
        /*1b60*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
        /*1b68*/                   DSETP.GT.AND P1, PT, |R14|, RZ, PT;                         /* 0x1a0e0000fce3dc81 */
        /*1b70*/              @!P1 DSETP.NEU.AND P0, PT, |R12|, +INF , PT;                     /* 0x1e8edffc00c1e481 */
        /*1b78*/              @!P1 SEL R9, R12, R14, P0;                                       /* 0x2000000038c26404 */
                                                                                               /* 0x22b202e202820277 */
        /*1b88*/              @!P1 SEL R12, R13, R15, P0;                                      /* 0x200000003cd32404 */
        /*1b90*/              @!P1 MOV R14, R9;                                                /* 0x280000002403a5e4 */
        /*1b98*/              @!P1 MOV R15, R12;                                               /* 0x280000003003e5e4 */
        /*1ba0*/                   DMUL R16, R10, R14;                                         /* 0x5000000038a41c01 */
        /*1ba8*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1bb0*/                   DMUL R16, R10, R12;                                         /* 0x5000000030a41c01 */
        /*1bb8*/                   BRK;                                                        /* 0xa800000000001de7 */
                                                                                               /* 0x22c292b2b20042e7 */
        /*1bc8*/                   RET;                                                        /* 0x9000000000001de7 */
        /*1bd0*/                   ISETP.LT.U32.AND P0, PT, R15, c[0x2][0x34], PT;             /* 0x188e4800d0f1dc03 */
        /*1bd8*/                   MOV R15, R19;                                               /* 0x280000004c03dde4 */
        /*1be0*/                   PBK 0x1d28;                                                 /* 0x6800000500001c07 */
        /*1be8*/               @P0 BRA 0x1c38;                                                 /* 0x40000001200001e7 */
        /*1bf0*/                   DFMA.RM R12, R12, R14, R16;                                 /* 0x20a0000038c31c01 */
        /*1bf8*/                   IADD32I R14.CC, R12, 0x1;                                   /* 0x0c00000004c39c02 */
                                                                                               /* 0x22d2e04202d29287 */
        /*1c08*/                   IADD.X R15, RZ, R13;                                        /* 0x4800000037f3dc43 */
        /*1c10*/                   DFMA.RP R10, -R12, R14, R10;                                /* 0x2114000038c29e01 */
        /*1c18*/                   DSETP.GT.AND P0, PT, R10, RZ, PT;                           /* 0x1a0e0000fca1dc01 */
        /*1c20*/                   SEL R12, R14, R12, P0;                                      /* 0x2000000030e31c04 */
        /*1c28*/                   SEL R13, R15, R13, P0;                                      /* 0x2000000034f35c04 */
        /*1c30*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1c38*/                   DSETP.EQU.AND P0, PT, R10, RZ, PT;                          /* 0x1d0e0000fca1dc01 */
                                                                                               /* 0x22b2c2b20042c2b7 */
        /*1c48*/               @P0 BRA 0x1d18;                                                 /* 0x40000003200001e7 */
        /*1c50*/                   ISETP.LT.AND P0, PT, R11, RZ, PT;                           /* 0x188e0000fcb1dc23 */
        /*1c58*/               @P0 MOV32I R12, 0x0;                                            /* 0x18000000000301e2 */
        /*1c60*/               @P0 MOV32I R13, 0xfff80000;                                     /* 0x1bffe000000341e2 */
        /*1c68*/               @P0 BRK;                                                        /* 0xa8000000000001e7 */
        /*1c70*/                   ISETP.GT.AND P0, PT, R11, c[0x2][0x38], PT;                 /* 0x1a0e4800e0b1dc23 */
        /*1c78*/               @P0 BRA 0x1d18;                                                 /* 0x40000002600001e7 */
                                                                                               /* 0x2292928042804207 */
        /*1c88*/                   DMUL R10, R10, 8.11296384146066816958e+31;                  /* 0x5000d1a400a29c01 */
        /*1c90*/                   MOV R14, RZ;                                                /* 0x28000000fc039de4 */
        /*1c98*/                   MOV32I R12, 0x0;                                            /* 0x1800000000031de2 */
        /*1ca0*/                   MUFU.RSQ64H R15, R11;                                       /* 0xc80000001cb3dc00 */
        /*1ca8*/                   MOV32I R13, 0x3fe00000;                                     /* 0x18ff800000035de2 */
        /*1cb0*/                   DMUL R16, R14, R14;                                         /* 0x5000000038e41c01 */
        /*1cb8*/                   DFMA R16, -R10, R16, c[0x2][0x0];                           /* 0x2020880000a41e01 */
                                                                                               /* 0x2292928202929237 */
        /*1cc8*/                   DFMA R12, R16, 0.375, R12;                                  /* 0x2018cff601031c01 */
        /*1cd0*/                   DMUL R16, R14, R16;                                         /* 0x5000000040e41c01 */
        /*1cd8*/                   DFMA R14, R12, R16, R14;                                    /* 0x201c000040c39c01 */
        /*1ce0*/                   DMUL R12, R10, R14;                                         /* 0x5000000038a31c01 */
        /*1ce8*/                   IADD32I R15, R15, -0x100000;                                /* 0x0bffc00000f3dc02 */
        /*1cf0*/                   DFMA R10, -R12, R12, R10;                                   /* 0x2014000030c29e01 */
        /*1cf8*/                   DFMA R12, R14, R10, R12;                                    /* 0x2018000028e31c01 */
                                                                                               /* 0x22e20042b202e047 */
        /*1d08*/                   IADD32I R13, R13, -0x3500000;                               /* 0x0bf2c00000d35c02 */
        /*1d10*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1d18*/                   DADD R12, R10, R10;                                         /* 0x4800000028a31c01 */
        /*1d20*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1d28*/                   MOV R28, R12;                                               /* 0x2800000030071de4 */
        /*1d30*/                   MOV R29, R13;                                               /* 0x2800000034075de4 */
        /*1d38*/                   RET;                                                        /* 0x9000000000001de7 */
        /*1d40*/                   BRA 0x1d40;                                                 /* 0x4003ffffe0001de7 */
        /*1d48*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*1d50*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*1d58*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*1d60*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*1d68*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*1d70*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*1d78*/                   NOP;                                                        /* 0x4000000000001de4 */
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
        /*00d8*/                   ST.E.64 [R2+0x4e358], RZ;               /* 0x9400138d602fdca5 */
        /*00e0*/                   ST.E.64 [R2+0x5dd98], RZ;               /* 0x94001776602fdca5 */
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
        /*0098*/                   LD.E.64 R8, [R2+0x10];                                     /* 0x8400000040221ca5 */
        /*00a0*/                   MOV32I R10, 0x1;                                           /* 0x1800000004029de2 */
        /*00a8*/                   LD.E.64 R4, [R2+0x4e350];                                  /* 0x8400138d40211ca5 */
        /*00b0*/                   MUFU.RCP64H R11, R9;                                       /* 0xc80000001892dc00 */
        /*00b8*/                   DFMA R6, -R8, R10, c[0x2][0x0];                            /* 0x2014880000819e01 */
                                                                                              /* 0x2282029292925237 */
        /*00c8*/                   DFMA R12, R6, R6, R6;                                      /* 0x200c000018631c01 */
        /*00d0*/                   DMUL R6, R4, c[0x0][0x148];                                /* 0x5000400520419c01 */
        /*00d8*/                   DFMA R4, R10, R12, R10;                                    /* 0x2014000030a11c01 */
        /*00e0*/                   DMUL R10, R6, R4;                                          /* 0x5000000010629c01 */
        /*00e8*/                   DFMA R14, -R8, R10, R6;                                    /* 0x200c000028839e01 */
        /*00f0*/                   DFMA R4, R4, R14, R10;                                     /* 0x2014000038411c01 */
        /*00f8*/                   FSET.GTU.AND RZ.CC, |R7|, c[0x2][0x8], PT;                 /* 0x160f4800207fdc80 */
                                                                                              /* 0x22b0420042b2c287 */
        /*0108*/                   FFMA R10, RZ, R9, R5;                                      /* 0x300a000027f29c00 */
        /*0110*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*0118*/               @P0 BRA CC.NEU, 0x160;                                         /* 0x40000001000001a7 */
        /*0120*/                   MOV R4, R6;                                                /* 0x2800000018011de4 */
        /*0128*/                   MOV R5, R7;                                                /* 0x280000001c015de4 */
        /*0130*/                   MOV R6, R8;                                                /* 0x2800000020019de4 */
        /*0138*/                   MOV R7, R9;                                                /* 0x280000002401dde4 */
                                                                                              /* 0x23f2323042b042b7 */
        /*0148*/                   CAL 0xa28;                                                 /* 0x5000002360011c07 */
        /*0150*/                   MOV R4, R8;                                                /* 0x2800000020011de4 */
        /*0158*/                   MOV R5, R9;                                                /* 0x2800000024015de4 */
        /*0160*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*0168*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*0170*/                   LD.E.64 R10, [R2+0x5dd90];                                 /* 0x8400177640229ca5 */
        /*0178*/                   LD.E.64 R8, [R2+0x2eed0];                                  /* 0x84000bbb40221ca5 */
                                                                                              /* 0x2292929252329287 */
        /*0188*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*0190*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*0198*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
        /*01a0*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*01a8*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*01b0*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*01b8*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
                                                                                              /* 0x22b2c27200429237 */
        /*01c8*/                   DFMA R18, R12, R14, R16;                                   /* 0x2020000038c49c01 */
        /*01d0*/                   DADD R4, R4, R8;                                           /* 0x4800000020411c01 */
        /*01d8*/                   FFMA R8, RZ, R7, R19;                                      /* 0x302600001ff21c00 */
        /*01e0*/                   ST.E.64 [R2+0x2eed0], R4;                                  /* 0x94000bbb40211ca5 */
        /*01e8*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*01f0*/                   FSETP.GT.AND P0, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003081dc80 */
        /*01f8*/               @P0 BRA CC.NEU, 0x230;                                         /* 0x40000000c00001a7 */
                                                                                              /* 0x223232b042b2b047 */
        /*0208*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*0210*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*0218*/                   CAL 0xa28;                                                 /* 0x5000002020011c07 */
        /*0220*/                   MOV R18, R8;                                               /* 0x2800000020049de4 */
        /*0228*/                   MOV R19, R9;                                               /* 0x280000002404dde4 */
        /*0230*/                   LD.E.64 R12, [R2+0x3e910];                                 /* 0x84000fa440231ca5 */
        /*0238*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
                                                                                              /* 0x22e2e04203f23237 */
        /*0248*/                   LD.E.64 R16, [R2+0x4e350];                                 /* 0x8400138d40241ca5 */
        /*0250*/                   LD.E.64 R4, [R2+0x2eed0];                                  /* 0x84000bbb40211ca5 */
        /*0258*/                   LD.E.64 R8, [R2+0xfa50];                                   /* 0x840003e940221ca5 */
        /*0260*/                   DADD R14, R18, R12;                                        /* 0x4800000031239c01 */
        /*0268*/                   LD.E.64 R12, [R2+0x1f490];                                 /* 0x840007d240231ca5 */
        /*0270*/                   MOV32I R18, 0x1;                                           /* 0x1800000004049de2 */
        /*0278*/                   ST.E.64 [R2+0x3e910], R14;                                 /* 0x94000fa440239ca5 */
                                                                                              /* 0x2232925232928047 */
        /*0288*/                   LD.E.64 R10, [R2+0x3e910];                                 /* 0x84000fa440229ca5 */
        /*0290*/                   MUFU.RCP64H R19, R7;                                       /* 0xc80000001874dc00 */
        /*0298*/                   DFMA R20, -R6, R18, c[0x2][0x0];                           /* 0x2024880000651e01 */
        /*02a0*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*02a8*/                   DMUL R16, R16, c[0x0][0x148];                              /* 0x5000400521041c01 */
        /*02b0*/                   DFMA R18, R18, R20, R18;                                   /* 0x2024000051249c01 */
        /*02b8*/                   DMUL R20, R16, R18;                                        /* 0x5000000049051c01 */
                                                                                              /* 0x228203f042029257 */
        /*02c8*/                   DFMA R4, R4, c[0x0][0x148], R8;                            /* 0x2010400520411c01 */
        /*02d0*/                   DFMA R8, -R6, R20, R16;                                    /* 0x2020000050621e01 */
        /*02d8*/                   DFMA R8, R18, R8, R20;                                     /* 0x2028000021221c01 */
        /*02e0*/                   ST.E.64 [R2+0xfa50], R4;                                   /* 0x940003e940211ca5 */
        /*02e8*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                /* 0x160f4800211fdc80 */
        /*02f0*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*02f8*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
                                                                                              /* 0x2232b2b042b2c047 */
        /*0308*/                   ST.E.64 [R2+0x1f490], R12;                                 /* 0x940007d240231ca5 */
        /*0310*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*0318*/               @P0 BRA CC.NEU, 0x338;                                         /* 0x40000000600001a7 */
        /*0320*/                   MOV R4, R16;                                               /* 0x2800000040011de4 */
        /*0328*/                   MOV R5, R17;                                               /* 0x2800000044015de4 */
        /*0330*/                   CAL 0xa28;                                                 /* 0x5000001bc0011c07 */
        /*0338*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
                                                                                              /* 0x2252329283f23047 */
        /*0348*/                   LD.E.64 R10, [R2+0x5dd90];                                 /* 0x8400177640229ca5 */
        /*0350*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*0358*/                   LD.E.64 R4, [R2+0x2eed0];                                  /* 0x84000bbb40211ca5 */
        /*0360*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*0368*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*0370*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
        /*0378*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
                                                                                              /* 0x2200429232929297 */
        /*0388*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*0390*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0398*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
        /*03a0*/                   DFMA R18, R12, R14, R16;                                   /* 0x2020000038c49c01 */
        /*03a8*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
        /*03b0*/                   FFMA R8, RZ, R7, R19;                                      /* 0x302600001ff21c00 */
        /*03b8*/                   ST.E.64 [R2+0x2eed0], R4;                                  /* 0x94000bbb40211ca5 */
                                                                                              /* 0x2202b2b042b2c277 */
        /*03c8*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*03d0*/                   FSETP.GT.AND P0, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003081dc80 */
        /*03d8*/               @P0 BRA CC.NEU, 0x410;                                         /* 0x40000000c00001a7 */
        /*03e0*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*03e8*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*03f0*/                   CAL 0xa28;                                                 /* 0x50000018c0011c07 */
        /*03f8*/                   MOV R18, R8;                                               /* 0x2800000020049de4 */
                                                                                              /* 0x2203f232323232b7 */
        /*0408*/                   MOV R19, R9;                                               /* 0x280000002404dde4 */
        /*0410*/                   LD.E.64 R12, [R2+0x3e910];                                 /* 0x84000fa440231ca5 */
        /*0418*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*0420*/                   LD.E.64 R16, [R2+0x4e350];                                 /* 0x8400138d40241ca5 */
        /*0428*/                   LD.E.64 R4, [R2+0x2eed0];                                  /* 0x84000bbb40211ca5 */
        /*0430*/                   LD.E.64 R8, [R2+0xfa50];                                   /* 0x840003e940221ca5 */
        /*0438*/                   DADD R14, R18, R12;                                        /* 0x4800000031239c01 */
                                                                                              /* 0x22329042e042e047 */
        /*0448*/                   LD.E.64 R12, [R2+0x1f490];                                 /* 0x840007d240231ca5 */
        /*0450*/                   MOV32I R18, 0x1;                                           /* 0x1800000004049de2 */
        /*0458*/                   ST.E.64 [R2+0x3e910], R14;                                 /* 0x94000fa440239ca5 */
        /*0460*/                   MUFU.RCP64H R19, R7;                                       /* 0xc80000001874dc00 */
        /*0468*/                   LD.E.64 R10, [R2+0x3e910];                                 /* 0x84000fa440229ca5 */
        /*0470*/                   DFMA R20, -R6, R18, c[0x2][0x0];                           /* 0x2024880000651e01 */
        /*0478*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
                                                                                              /* 0x2202029252329257 */
        /*0488*/                   DMUL R16, R16, c[0x0][0x148];                              /* 0x5000400521041c01 */
        /*0490*/                   DFMA R18, R18, R20, R18;                                   /* 0x2024000051249c01 */
        /*0498*/                   DMUL R20, R16, R18;                                        /* 0x5000000049051c01 */
        /*04a0*/                   DFMA R4, R4, c[0x0][0x148], R8;                            /* 0x2010400520411c01 */
        /*04a8*/                   DFMA R8, -R6, R20, R16;                                    /* 0x2020000050621e01 */
        /*04b0*/                   DFMA R8, R18, R8, R20;                                     /* 0x2028000021221c01 */
        /*04b8*/                   ST.E.64 [R2+0xfa50], R4;                                   /* 0x940003e940211ca5 */
                                                                                              /* 0x2202b2c0428203f7 */
        /*04c8*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                /* 0x160f4800211fdc80 */
        /*04d0*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*04d8*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*04e0*/                   ST.E.64 [R2+0x1f490], R12;                                 /* 0x940007d240231ca5 */
        /*04e8*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*04f0*/               @P0 BRA CC.NEU, 0x518;                                         /* 0x40000000800001a7 */
        /*04f8*/                   MOV R4, R16;                                               /* 0x2800000040011de4 */
                                                                                              /* 0x2283f2323042b2b7 */
        /*0508*/                   MOV R5, R17;                                               /* 0x2800000044015de4 */
        /*0510*/                   CAL 0xa28;                                                 /* 0x5000001440011c07 */
        /*0518*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*0520*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*0528*/                   LD.E.64 R10, [R2+0x5dd90];                                 /* 0x8400177640229ca5 */
        /*0530*/                   LD.E.64 R4, [R2+0x2eed0];                                  /* 0x84000bbb40211ca5 */
        /*0538*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
                                                                                              /* 0x2232929292523297 */
        /*0548*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*0550*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
        /*0558*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*0560*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*0568*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0570*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
        /*0578*/                   DFMA R18, R12, R14, R16;                                   /* 0x2020000038c49c01 */
                                                                                              /* 0x2202b2c272004297 */
        /*0588*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
        /*0590*/                   FFMA R8, RZ, R7, R19;                                      /* 0x302600001ff21c00 */
        /*0598*/                   ST.E.64 [R2+0x2eed0], R4;                                  /* 0x94000bbb40211ca5 */
        /*05a0*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*05a8*/                   FSETP.GT.AND P0, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003081dc80 */
        /*05b0*/               @P0 BRA CC.NEU, 0x5e8;                                         /* 0x40000000c00001a7 */
        /*05b8*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
                                                                                              /* 0x22323232b042b2b7 */
        /*05c8*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*05d0*/                   CAL 0xa28;                                                 /* 0x5000001140011c07 */
        /*05d8*/                   MOV R18, R8;                                               /* 0x2800000020049de4 */
        /*05e0*/                   MOV R19, R9;                                               /* 0x280000002404dde4 */
        /*05e8*/                   LD.E.64 R12, [R2+0x3e910];                                 /* 0x84000fa440231ca5 */
        /*05f0*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*05f8*/                   LD.E.64 R16, [R2+0x4e350];                                 /* 0x8400138d40241ca5 */
                                                                                              /* 0x22e042e04203f237 */
        /*0608*/                   LD.E.64 R4, [R2+0x2eed0];                                  /* 0x84000bbb40211ca5 */
        /*0610*/                   LD.E.64 R8, [R2+0xfa50];                                   /* 0x840003e940221ca5 */
        /*0618*/                   DADD R14, R18, R12;                                        /* 0x4800000031239c01 */
        /*0620*/                   LD.E.64 R12, [R2+0x1f490];                                 /* 0x840007d240231ca5 */
        /*0628*/                   MOV32I R18, 0x1;                                           /* 0x1800000004049de2 */
        /*0630*/                   ST.E.64 [R2+0x3e910], R14;                                 /* 0x94000fa440239ca5 */
        /*0638*/                   MUFU.RCP64H R19, R7;                                       /* 0xc80000001874dc00 */
                                                                                              /* 0x2252329252329047 */
        /*0648*/                   LD.E.64 R10, [R2+0x3e910];                                 /* 0x84000fa440229ca5 */
        /*0650*/                   DFMA R20, -R6, R18, c[0x2][0x0];                           /* 0x2024880000651e01 */
        /*0658*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*0660*/                   DMUL R16, R16, c[0x0][0x148];                              /* 0x5000400521041c01 */
        /*0668*/                   DFMA R18, R18, R20, R18;                                   /* 0x2024000051249c01 */
        /*0670*/                   DMUL R20, R16, R18;                                        /* 0x5000000049051c01 */
        /*0678*/                   DFMA R4, R4, c[0x0][0x148], R8;                            /* 0x2010400520411c01 */
                                                                                              /* 0x22028203f0420297 */
        /*0688*/                   DFMA R8, -R6, R20, R16;                                    /* 0x2020000050621e01 */
        /*0690*/                   DFMA R8, R18, R8, R20;                                     /* 0x2028000021221c01 */
        /*0698*/                   ST.E.64 [R2+0xfa50], R4;                                   /* 0x940003e940211ca5 */
        /*06a0*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                /* 0x160f4800211fdc80 */
        /*06a8*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*06b0*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*06b8*/                   ST.E.64 [R2+0x1f490], R12;                                 /* 0x940007d240231ca5 */
                                                                                              /* 0x223042b2b042b2c7 */
        /*06c8*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*06d0*/               @P0 BRA CC.NEU, 0x6f0;                                         /* 0x40000000600001a7 */
        /*06d8*/                   MOV R4, R16;                                               /* 0x2800000040011de4 */
        /*06e0*/                   MOV R5, R17;                                               /* 0x2800000044015de4 */
        /*06e8*/                   CAL 0xa28;                                                 /* 0x5000000ce0011c07 */
        /*06f0*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*06f8*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
                                                                                              /* 0x229252329283f237 */
        /*0708*/                   LD.E.64 R10, [R2+0x5dd90];                                 /* 0x8400177640229ca5 */
        /*0710*/                   LD.E.64 R4, [R2+0x2eed0];                                  /* 0x84000bbb40211ca5 */
        /*0718*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*0720*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
        /*0728*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
        /*0730*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*0738*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
                                                                                              /* 0x2272004292329297 */
        /*0748*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0750*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
        /*0758*/                   DFMA R18, R12, R14, R16;                                   /* 0x2020000038c49c01 */
        /*0760*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
        /*0768*/                   FFMA R8, RZ, R7, R19;                                      /* 0x302600001ff21c00 */
        /*0770*/                   ST.E.64 [R2+0x2eed0], R4;                                  /* 0x94000bbb40211ca5 */
        /*0778*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
                                                                                              /* 0x22b042b2b042b2c7 */
        /*0788*/                   FSETP.GT.AND P0, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003081dc80 */
        /*0790*/               @P0 BRA CC.NEU, 0x7c0;                                         /* 0x40000000a00001a7 */
        /*0798*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*07a0*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
        /*07a8*/                   CAL 0xa28;                                                 /* 0x50000009e0011c07 */
        /*07b0*/                   MOV R18, R8;                                               /* 0x2800000020049de4 */
        /*07b8*/                   MOV R19, R9;                                               /* 0x280000002404dde4 */
                                                                                              /* 0x22e203f232323237 */
        /*07c8*/                   LD.E.64 R12, [R2+0x3e910];                                 /* 0x84000fa440231ca5 */
        /*07d0*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*07d8*/                   LD.E.64 R16, [R2+0x4e350];                                 /* 0x8400138d40241ca5 */
        /*07e0*/                   LD.E.64 R4, [R2+0x2eed0];                                  /* 0x84000bbb40211ca5 */
        /*07e8*/                   LD.E.64 R8, [R2+0xfa50];                                   /* 0x840003e940221ca5 */
        /*07f0*/                   DADD R14, R18, R12;                                        /* 0x4800000031239c01 */
        /*07f8*/                   LD.E.64 R12, [R2+0x1f490];                                 /* 0x840007d240231ca5 */
                                                                                              /* 0x229280423042e047 */
        /*0808*/                   ST.E.64 [R2+0x3e910], R14;                                 /* 0x94000fa440239ca5 */
        /*0810*/                   MUFU.RCP64H R19, R7;                                       /* 0xc80000001874dc00 */
        /*0818*/                   LD.E.64 R10, [R2+0x3e910];                                 /* 0x84000fa440229ca5 */
        /*0820*/                   DMUL R16, R16, c[0x0][0x148];                              /* 0x5000400521041c01 */
        /*0828*/                   MOV32I R18, 0x1;                                           /* 0x1800000004049de2 */
        /*0830*/                   DFMA R4, R4, c[0x0][0x148], R8;                            /* 0x2010400520411c01 */
        /*0838*/                   DFMA R20, -R6, R18, c[0x2][0x0];                           /* 0x2024880000651e01 */
                                                                                              /* 0x23f0420292929297 */
        /*0848*/                   DFMA R20, R20, R20, R20;                                   /* 0x2028000051451c01 */
        /*0850*/                   DFMA R18, R18, R20, R18;                                   /* 0x2024000051249c01 */
        /*0858*/                   DMUL R20, R16, R18;                                        /* 0x5000000049051c01 */
        /*0860*/                   DFMA R8, -R6, R20, R16;                                    /* 0x2020000050621e01 */
        /*0868*/                   DFMA R8, R18, R8, R20;                                     /* 0x2028000021221c01 */
        /*0870*/                   ST.E.64 [R2+0xfa50], R4;                                   /* 0x940003e940211ca5 */
        /*0878*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                /* 0x160f4800211fdc80 */
                                                                                              /* 0x22b042b2c0428207 */
        /*0888*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*0890*/                   FFMA R10, RZ, R7, R9;                                      /* 0x301200001ff29c00 */
        /*0898*/                   ST.E.64 [R2+0x1f490], R12;                                 /* 0x940007d240231ca5 */
        /*08a0*/                   FSETP.GT.AND P0, PT, |R10|, c[0x2][0xc], PT;               /* 0x220e480030a1dc80 */
        /*08a8*/               @P0 BRA CC.NEU, 0x8d0;                                         /* 0x40000000800001a7 */
        /*08b0*/                   MOV R4, R16;                                               /* 0x2800000040011de4 */
        /*08b8*/                   MOV R5, R17;                                               /* 0x2800000044015de4 */
                                                                                              /* 0x229283f2323042b7 */
        /*08c8*/                   CAL 0xa28;                                                 /* 0x5000000560011c07 */
        /*08d0*/                   LD.E.64 R6, [R2+0x10];                                     /* 0x8400000040219ca5 */
        /*08d8*/                   MOV32I R12, 0x1;                                           /* 0x1800000004031de2 */
        /*08e0*/                   LD.E.64 R10, [R2+0x5dd90];                                 /* 0x8400177640229ca5 */
        /*08e8*/                   LD.E.64 R4, [R2+0x2eed0];                                  /* 0x84000bbb40211ca5 */
        /*08f0*/                   MUFU.RCP64H R13, R7;                                       /* 0xc800000018735c00 */
        /*08f8*/                   DFMA R14, -R6, R12, c[0x2][0x0];                           /* 0x2018880000639e01 */
                                                                                              /* 0x2292329292925237 */
        /*0908*/                   DFMA R14, R14, R14, R14;                                   /* 0x201c000038e39c01 */
        /*0910*/                   DMUL R10, R10, c[0x0][0x148];                              /* 0x5000400520a29c01 */
        /*0918*/                   DFMA R12, R12, R14, R12;                                   /* 0x2018000038c31c01 */
        /*0920*/                   DMUL R16, R10, R12;                                        /* 0x5000000030a41c01 */
        /*0928*/                   DFMA R14, -R6, R16, R10;                                   /* 0x2014000040639e01 */
        /*0930*/                   DFMA R16, R12, R14, R16;                                   /* 0x2020000038c41c01 */
        /*0938*/                   DADD R4, R8, R4;                                           /* 0x4800000010811c01 */
                                                                                              /* 0x22b042b2c2720047 */
        /*0948*/                   FFMA R8, RZ, R7, R17;                                      /* 0x302200001ff21c00 */
        /*0950*/                   ST.E.64 [R2+0x2eed0], R4;                                  /* 0x94000bbb40211ca5 */
        /*0958*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0x8], PT;                /* 0x160f480020bfdc80 */
        /*0960*/                   FSETP.GT.AND P0, PT, |R8|, c[0x2][0xc], PT;                /* 0x220e48003081dc80 */
        /*0968*/               @P0 BRA CC.NEU, 0x9a0;                                         /* 0x40000000c00001a7 */
        /*0970*/                   MOV R4, R10;                                               /* 0x2800000028011de4 */
        /*0978*/                   MOV R5, R11;                                               /* 0x280000002c015de4 */
                                                                                              /* 0x2200428042b042b7 */
        /*0988*/                   CAL 0xa28;                                                 /* 0x5000000260011c07 */
        /*0990*/                   MOV R16, R8;                                               /* 0x2800000020041de4 */
        /*0998*/                   MOV R17, R9;                                               /* 0x2800000024045de4 */
        /*09a0*/                   LD.E.64 R4, [R2+0x3e910];                                  /* 0x84000fa440211ca5 */
        /*09a8*/                   IADD32I R0, R0, 0x5;                                       /* 0x0800000014001c02 */
        /*09b0*/                   LD.E.64 R6, [R2+0x2eed0];                                  /* 0x84000bbb40219ca5 */
        /*09b8*/                   ISETP.NE.AND P0, PT, R0, RZ, PT;                           /* 0x1a8e0000fc01dc23 */
                                                                                              /* 0x23f2e2e04293f237 */
        /*09c8*/                   LD.E.64 R8, [R2+0xfa50];                                   /* 0x840003e940221ca5 */
        /*09d0*/                   LD.E.64 R12, [R2+0x1f490];                                 /* 0x840007d240231ca5 */
        /*09d8*/                   DADD R4, R16, R4;                                          /* 0x4800000011011c01 */
        /*09e0*/                   ST.E.64 [R2+0x3e910], R4;                                  /* 0x94000fa440211ca5 */
        /*09e8*/                   DFMA R6, R6, c[0x0][0x148], R8;                            /* 0x2010400520619c01 */
        /*09f0*/                   LD.E.64 R10, [R2+0x3e910];                                 /* 0x84000fa440229ca5 */
        /*09f8*/                   ST.E.64 [R2+0xfa50], R6;                                   /* 0x940003e940219ca5 */
                                                                                              /* 0x22028042e2e04297 */
        /*0a08*/                   DFMA R12, R10, c[0x0][0x148], R12;                         /* 0x2018400520a31c01 */
        /*0a10*/                   ST.E.64 [R2+0x1f490], R12;                                 /* 0x940007d240231ca5 */
        /*0a18*/               @P0 BRA 0x98;                                                  /* 0x4003ffd9e00001e7 */
        /*0a20*/                   EXIT;                                                      /* 0x8000000000001de7 */
        /*0a28*/                   LOP32I.AND R8, R7, 0x40000000;                             /* 0x3900000000721c02 */
        /*0a30*/                   MOV32I R12, 0x5ff00000;                                    /* 0x197fc00000031de2 */
        /*0a38*/                   ISETP.LT.U32.AND P0, PT, R8, c[0x2][0x10], PT;             /* 0x188e48004081dc03 */
                                                                                              /* 0x22c042028042b047 */
        /*0a48*/                   MOV R8, RZ;                                                /* 0x28000000fc021de4 */
        /*0a50*/                   LOP32I.AND R13, R5, 0x7f800000;                            /* 0x39fe000000535c02 */
        /*0a58*/                   SEL R9, R12, c[0x2][0x14], P0;                             /* 0x2000480050c25c04 */
        /*0a60*/                   MOV32I R14, 0x1;                                           /* 0x1800000004039de2 */
        /*0a68*/                   DMUL R10, R6, R8;                                          /* 0x5000000020629c01 */
        /*0a70*/                   ISETP.LT.U32.AND P0, PT, R13, c[0x2][0x18], PT;            /* 0x188e480060d1dc03 */
        /*0a78*/                   MOV R16, RZ;                                               /* 0x28000000fc041de4 */
                                                                                              /* 0x2292925232928047 */
        /*0a88*/                   MUFU.RCP64H R15, R11;                                      /* 0xc800000018b3dc00 */
        /*0a90*/                   SEL R17, R12, c[0x2][0x14], P0;                            /* 0x2000480050c45c04 */
        /*0a98*/                   DFMA R18, -R10, R14, c[0x2][0x0];                          /* 0x201c880000a49e01 */
        /*0aa0*/                   DFMA R18, R18, R18, R18;                                   /* 0x2024000049249c01 */
        /*0aa8*/                   DMUL R16, R4, R16;                                         /* 0x5000000040441c01 */
        /*0ab0*/                   DFMA R14, R14, R18, R14;                                   /* 0x201c000048e39c01 */
        /*0ab8*/                   DMUL R20, R16, R14;                                        /* 0x5000000039051c01 */
                                                                                              /* 0x2202c042b2d29297 */
        /*0ac8*/                   DFMA R10, -R10, R20, R16;                                  /* 0x2020000050a29e01 */
        /*0ad0*/                   DFMA R14, R14, R10, R20;                                   /* 0x2028000028e39c01 */
        /*0ad8*/                   DSETP.LEU.AND P0, PT, |R14|, RZ, PT;                       /* 0x1d8e0000fce1dc81 */
        /*0ae0*/               @P0 BRA 0xc98;                                                 /* 0x40000006c00001e7 */
        /*0ae8*/                   ISETP.GT.U32.AND P0, PT, R13, c[0x2][0x1c], PT;            /* 0x1a0e480070d1dc03 */
        /*0af0*/                   MOV R10, RZ;                                               /* 0x28000000fc029de4 */
        /*0af8*/                   SEL R11, R12, c[0x2][0x14], P0;                            /* 0x2000480050c2dc04 */
                                                                                              /* 0x22d2925232523277 */
        /*0b08*/                   DMUL R12, R8, R14;                                         /* 0x5000000038831c01 */
        /*0b10*/                   DMUL R14, R14, R10;                                        /* 0x5000000028e39c01 */
        /*0b18*/                   DMUL R10, R10, R12;                                        /* 0x5000000030a29c01 */
        /*0b20*/                   DMUL R14, R8, R14;                                         /* 0x5000000038839c01 */
        /*0b28*/                   DFMA R8, R6, R10, -R4;                                     /* 0x2008000028621d01 */
        /*0b30*/                   DFMA R12, R6, R14, -R4;                                    /* 0x2008000038631d01 */
        /*0b38*/                   DSETP.GT.AND P0, PT, |R8|, |R12|, PT;                      /* 0x1a0e00003081dcc1 */
                                                                                              /* 0x22220042e2c27207 */
        /*0b48*/                   SEL R9, R15, R11, P0;                                      /* 0x200000002cf25c04 */
        /*0b50*/                   SEL R8, R14, R10, P0;                                      /* 0x2000000028e21c04 */
        /*0b58*/                   FSETP.GTU.AND P0, PT, |R9|, 1.469367938527859385e-39, PT;  /* 0x260ec0040091dc80 */
        /*0b60*/               @P0 RET;                                                       /* 0x90000000000001e7 */
        /*0b68*/                   FSETP.LT.AND P0, PT, |R5|, 1.5046327690525280102e-36, PT;  /* 0x208ec1000051dc80 */
        /*0b70*/                   LOP32I.AND R14, R8, 0xfffffffe;                            /* 0x3bfffffff8839c02 */
        /*0b78*/                   MOV32I R11, 0x58500000;                                    /* 0x196140000002dde2 */
                                                                                              /* 0x2202023280428047 */
        /*0b88*/                   MOV R10, RZ;                                               /* 0x28000000fc029de4 */
        /*0b90*/                   LOP32I.OR R15, R8, 0x1;                                    /* 0x380000000483dc42 */
        /*0b98*/                   SEL R11, R11, c[0x2][0x4], P0;                             /* 0x2000480010b2dc04 */
        /*0ba0*/                   MOV R8, R14;                                               /* 0x2800000038021de4 */
        /*0ba8*/                   DMUL R6, R6, R10;                                          /* 0x5000000028619c01 */
        /*0bb0*/                   DMUL R4, R4, R10;                                          /* 0x5000000028411c01 */
        /*0bb8*/                   MOV R10, R15;                                              /* 0x280000003c029de4 */
                                                                                              /* 0x22c04282d2923277 */
        /*0bc8*/                   MOV R11, R9;                                               /* 0x280000002402dde4 */
        /*0bd0*/                   DFMA R12, R8, R6, -R4;                                     /* 0x2008000018831d01 */
        /*0bd8*/                   DFMA R10, R10, R6, -R4;                                    /* 0x2008000018a29d01 */
        /*0be0*/                   DSETP.GT.AND P0, PT, |R12|, |R10|, PT;                     /* 0x1a0e000028c1dcc1 */
        /*0be8*/                   SEL R15, R15, R14, P0;                                     /* 0x2000000038f3dc04 */
        /*0bf0*/                   IADD32I R12.CC, R15, 0x1;                                  /* 0x0c00000004f31c02 */
        /*0bf8*/                   LOP32I.AND R8, R15, 0x1;                                   /* 0x3800000004f21c02 */
                                                                                              /* 0x2262020042c20047 */
        /*0c08*/                   ISETP.EQ.U32.AND P0, PT, R8, 0x1, PT;                      /* 0x190ec0000481dc03 */
        /*0c10*/                   IADD.X R13, RZ, R9;                                        /* 0x4800000027f35c43 */
        /*0c18*/                   IADD32I R10.CC, R15, -0x1;                                 /* 0x0ffffffffcf29c02 */
        /*0c20*/                   SEL R12, R15, R12, !P0;                                    /* 0x2010000030f31c04 */
        /*0c28*/                   IADD32I.X R8, R9, -0x1;                                    /* 0x0bfffffffc921c42 */
        /*0c30*/                   SEL R13, R9, R13, !P0;                                     /* 0x2010000034935c04 */
        /*0c38*/                   SEL R10, R10, R15, !P0;                                    /* 0x201000003ca29c04 */
                                                                                              /* 0x2202d29232820207 */
        /*0c48*/                   SEL R11, R8, R9, !P0;                                      /* 0x201000002482dc04 */
        /*0c50*/                   MOV R8, R12;                                               /* 0x2800000030021de4 */
        /*0c58*/                   MOV R9, R13;                                               /* 0x2800000034025de4 */
        /*0c60*/                   DFMA R8, R6, R8, -R4;                                      /* 0x2008000020621d01 */
        /*0c68*/                   DFMA R4, R6, R10, -R4;                                     /* 0x2008000028611d01 */
        /*0c70*/                   DSETP.GT.AND P0, PT, |R8|, |R4|, PT;                       /* 0x1a0e00001081dcc1 */
        /*0c78*/                   SEL R8, R10, R12, P0;                                      /* 0x2000000030a21c04 */
                                                                                              /* 0x22d28042b2d2e047 */
        /*0c88*/                   SEL R9, R11, R13, P0;                                      /* 0x2000000034b25c04 */
        /*0c90*/                   RET;                                                       /* 0x9000000000001de7 */
        /*0c98*/                   DSETP.EQ.AND P0, PT, R14, RZ, PT;                          /* 0x190e0000fce1dc01 */
        /*0ca0*/               @P0 BRA 0xd00;                                                 /* 0x40000001600001e7 */
        /*0ca8*/                   MUFU.RCP64H R9, R7;                                        /* 0xc800000018725c00 */
        /*0cb0*/                   MOV R8, RZ;                                                /* 0x28000000fc021de4 */
        /*0cb8*/                   DSETP.GT.AND P1, PT, |R8|, RZ, PT;                         /* 0x1a0e0000fc83dc81 */
                                                                                              /* 0x22e20282027202d7 */
        /*0cc8*/              @!P1 DSETP.NEU.AND P0, PT, |R6|, +INF , PT;                     /* 0x1e8edffc0061e481 */
        /*0cd0*/              @!P1 SEL R6, R6, R8, P0;                                        /* 0x200000002061a404 */
        /*0cd8*/              @!P1 SEL R7, R7, R9, P0;                                        /* 0x200000002471e404 */
        /*0ce0*/              @!P1 MOV R8, R6;                                                /* 0x28000000180225e4 */
        /*0ce8*/              @!P1 MOV R9, R7;                                                /* 0x280000001c0265e4 */
        /*0cf0*/                   DMUL R8, R4, R8;                                           /* 0x5000000020421c01 */
        /*0cf8*/                   RET;                                                       /* 0x9000000000001de7 */
                                                                                              /* 0x200000000002e207 */
        /*0d08*/                   DMUL R8, R4, R6;                                           /* 0x5000000018421c01 */
        /*0d10*/                   RET;                                                       /* 0x9000000000001de7 */
        /*0d18*/                   BRA 0xd18;                                                 /* 0x4003ffffe0001de7 */
        /*0d20*/                   NOP;                                                       /* 0x4000000000001de4 */
        /*0d28*/                   NOP;                                                       /* 0x4000000000001de4 */
        /*0d30*/                   NOP;                                                       /* 0x4000000000001de4 */
        /*0d38*/                   NOP;                                                       /* 0x4000000000001de4 */
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
        /*0088*/                   ST.E.64 [R2+0x4e350], RZ;         /* 0x9400138d402fdca5 */
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
        /*0090*/                   ST.E.64 [R2+0x4e350], RZ;         /* 0x9400138d402fdca5 */
        /*0098*/                   ST.E.64 [R2+0x5dd90], RZ;         /* 0x94001776402fdca5 */
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
        /*00a0*/                   LD.E.64 R4, [R2+0x4e350];                                   /* 0x8400138d40211ca5 */
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
        /*0158*/                   LD.E.64 R16, [R2+0x5dd90];                                  /* 0x8400177640241ca5 */
        /*0160*/                   LD.E.64 R14, [R2+0x2eed0];                                  /* 0x84000bbb40239ca5 */
        /*0168*/                   LD.E.64 R4, [R2+0x3e910];                                   /* 0x84000fa440211ca5 */
        /*0170*/                   LD.E.64 R6, [R2+0xfa50];                                    /* 0x840003e940219ca5 */
        /*0178*/                   LD.E.64 R8, [R2+0x1f490];                                   /* 0x840007d240221ca5 */
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
        /*01e0*/                   ST.E.64 [R2+0x2eed0], R12;                                  /* 0x94000bbb40231ca5 */
        /*01e8*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                 /* 0x160f4800211fdc80 */
        /*01f0*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0xc], PT;                 /* 0x220e48003001dc80 */
        /*01f8*/               @P0 BRA CC.NEU, 0x220;                                          /* 0x40000000800001a7 */
                                                                                               /* 0x22e2e25232b2b047 */
        /*0208*/                   MOV R14, R16;                                               /* 0x2800000040039de4 */
        /*0210*/                   MOV R15, R17;                                               /* 0x280000004403dde4 */
        /*0218*/                   CAL 0x2e8;                                                  /* 0x5000000320011c07 */
        /*0220*/                   DFMA R6, R12, c[0x0][0x148], R6;                            /* 0x200c400520c19c01 */
        /*0228*/                   DADD R4, R4, R22;                                           /* 0x4800000058411c01 */
        /*0230*/                   ST.E.64 [R2+0xfa50], R6;                                    /* 0x940003e940219ca5 */
        /*0238*/                   LD.E.64 R12, [R2+0xfa50];                                   /* 0x840003e940231ca5 */
                                                                                               /* 0x22b04292f2d3f2e7 */
        /*0248*/                   ST.E.64 [R2+0x3e910], R4;                                   /* 0x94000fa440211ca5 */
        /*0250*/                   LD.E.64 R10, [R2+0x3e910];                                  /* 0x84000fa440229ca5 */
        /*0258*/                   DSETP.GT.AND P0, PT, R12, 1, PT;                            /* 0x1a0ecffc00c1dc01 */
        /*0260*/                   DSETP.LT.OR P0, PT, R12, -1, P0;                            /* 0x18a0effc00c1dc01 */
        /*0268*/                   DFMA R8, R10, c[0x0][0x148], R8;                            /* 0x2010400520a21c01 */
        /*0270*/                   ST.E.64 [R2+0x1f490], R8;                                   /* 0x940007d240221ca5 */
        /*0278*/              @!P0 BRA 0x2a8;                                                  /* 0x40000000a00021e7 */
                                                                                               /* 0x22e2d2d2b283f237 */
        /*0288*/                   LD.E.64 R4, [R2+0x2eed0];                                   /* 0x84000bbb40211ca5 */
        /*0290*/                   LD.E.64 R8, [R2+0x1f490];                                   /* 0x840007d240221ca5 */
        /*0298*/                   F2F.F64.F64 R4, -R4;                                        /* 0x1000000011b11d04 */
        /*02a0*/                   ST.E.64 [R2+0x2eed0], R4;                                   /* 0x94000bbb40211ca5 */
        /*02a8*/                   DSETP.GT.AND P0, PT, R8, 1, PT;                             /* 0x1a0ecffc0081dc01 */
        /*02b0*/                   DSETP.LT.OR P0, PT, R8, -1, P0;                             /* 0x18a0effc0081dc01 */
        /*02b8*/              @!P0 EXIT;                                                       /* 0x80000000000021e7 */
                                                                                               /* 0x22028042e04283f7 */
        /*02c8*/                   LD.E.64 R4, [R2+0x3e910];                                   /* 0x84000fa440211ca5 */
        /*02d0*/                   F2F.F64.F64 R4, -R4;                                        /* 0x1000000011b11d04 */
        /*02d8*/                   ST.E.64 [R2+0x3e910], R4;                                   /* 0x94000fa440211ca5 */
        /*02e0*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*02e8*/                   LOP32I.AND R0, R11, 0x40000000;                             /* 0x3900000000b01c02 */
        /*02f0*/                   LOP32I.AND R26, R15, 0x7f800000;                            /* 0x39fe000000f69c02 */
        /*02f8*/                   ISETP.LT.U32.AND P0, PT, R0, c[0x2][0x10], PT;              /* 0x188e48004001dc03 */
                                                                                               /* 0x22c042028042b047 */
        /*0308*/                   MOV32I R0, 0x5ff00000;                                      /* 0x197fc00000001de2 */
        /*0310*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*0318*/                   SEL R17, R0, c[0x2][0x14], P0;                              /* 0x2000480050045c04 */
        /*0320*/                   MOV32I R20, 0x1;                                            /* 0x1800000004051de2 */
        /*0328*/                   DMUL R18, R10, R16;                                         /* 0x5000000040a49c01 */
        /*0330*/                   ISETP.LT.U32.AND P0, PT, R26, c[0x2][0x18], PT;             /* 0x188e480061a1dc03 */
        /*0338*/                   MOV R24, RZ;                                                /* 0x28000000fc061de4 */
                                                                                               /* 0x2292925232928047 */
        /*0348*/                   MUFU.RCP64H R21, R19;                                       /* 0xc800000019355c00 */
        /*0350*/                   SEL R25, R0, c[0x2][0x14], P0;                              /* 0x2000480050065c04 */
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
        /*03c8*/                   SEL R19, R0, c[0x2][0x14], P0;                              /* 0x200048005004dc04 */
        /*03d0*/                   DMUL R22, R22, R18;                                         /* 0x5000000049659c01 */
        /*03d8*/                   DMUL R20, R18, R20;                                         /* 0x5000000051251c01 */
        /*03e0*/                   DMUL R16, R16, R22;                                         /* 0x5000000059041c01 */
        /*03e8*/                   DFMA R18, R10, R20, -R14;                                   /* 0x201c000050a49d01 */
        /*03f0*/                   DFMA R22, R10, R16, -R14;                                   /* 0x201c000040a59d01 */
        /*03f8*/                   DSETP.GT.AND P0, PT, |R18|, |R22|, PT;                      /* 0x1a0e00005921dcc1 */
                                                                                               /* 0x220042b042c27207 */
        /*0408*/                   SEL R17, R17, R21, P0;                                      /* 0x2000000055145c04 */
        /*0410*/                   SEL R18, R16, R20, P0;                                      /* 0x2000000051049c04 */
        /*0418*/                   FSETP.GTU.AND P0, PT, |R17|, 1.469367938527859385e-39, PT;  /* 0x260ec0040111dc80 */
        /*0420*/                   MOV R19, R17;                                               /* 0x280000004404dde4 */
        /*0428*/               @P0 BRA 0x5d0;                                                  /* 0x40000006800001e7 */
        /*0430*/                   FSETP.LT.AND P0, PT, |R15|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000f1dc80 */
        /*0438*/                   LOP32I.OR R22, R18, 0x1;                                    /* 0x3800000005259c42 */
                                                                                               /* 0x2202023282a20047 */
        /*0448*/                   MOV32I R21, 0x58500000;                                     /* 0x1961400000055de2 */
        /*0450*/                   MOV R20, RZ;                                                /* 0x28000000fc051de4 */
        /*0458*/                   LOP32I.AND R16, R18, 0xfffffffe;                            /* 0x3bfffffff9241c02 */
        /*0460*/                   SEL R21, R21, c[0x2][0x4], P0;                              /* 0x2000480011555c04 */
        /*0468*/                   DMUL R10, R10, R20;                                         /* 0x5000000050a29c01 */
        /*0470*/                   DMUL R14, R14, R20;                                         /* 0x5000000050e39c01 */
        /*0478*/                   MOV R20, R22;                                               /* 0x2800000058051de4 */
                                                                                               /* 0x22c04282d2923277 */
        /*0488*/                   MOV R21, R17;                                               /* 0x2800000044055de4 */
        /*0490*/                   DFMA R18, R16, R10, -R14;                                   /* 0x201c000029049d01 */
        /*0498*/                   DFMA R20, R20, R10, -R14;                                   /* 0x201c000029451d01 */
        /*04a0*/                   DSETP.GT.AND P0, PT, |R18|, |R20|, PT;                      /* 0x1a0e00005121dcc1 */
        /*04a8*/                   SEL R23, R22, R16, P0;                                      /* 0x200000004165dc04 */
        /*04b0*/                   IADD32I R18.CC, R23, 0x1;                                   /* 0x0c00000005749c02 */
        /*04b8*/                   LOP32I.AND R16, R23, 0x1;                                   /* 0x3800000005741c02 */
                                                                                               /* 0x2262020042c20047 */
        /*04c8*/                   ISETP.EQ.U32.AND P0, PT, R16, 0x1, PT;                      /* 0x190ec0000501dc03 */
        /*04d0*/                   IADD.X R0, RZ, R17;                                         /* 0x4800000047f01c43 */
        /*04d8*/                   IADD32I R20.CC, R23, -0x1;                                  /* 0x0ffffffffd751c02 */
        /*04e0*/                   SEL R18, R23, R18, !P0;                                     /* 0x2010000049749c04 */
        /*04e8*/                   IADD32I.X R21, R17, -0x1;                                   /* 0x0bfffffffd155c42 */
        /*04f0*/                   SEL R19, R17, R0, !P0;                                      /* 0x201000000114dc04 */
        /*04f8*/                   SEL R20, R20, R23, !P0;                                     /* 0x201000005d451c04 */
                                                                                               /* 0x2202d29232820207 */
        /*0508*/                   SEL R21, R21, R17, !P0;                                     /* 0x2010000045555c04 */
        /*0510*/                   MOV R16, R18;                                               /* 0x2800000048041de4 */
        /*0518*/                   MOV R17, R19;                                               /* 0x280000004c045de4 */
        /*0520*/                   DFMA R16, R10, R16, -R14;                                   /* 0x201c000040a41d01 */
        /*0528*/                   DFMA R10, R10, R20, -R14;                                   /* 0x201c000050a29d01 */
        /*0530*/                   DSETP.GT.AND P0, PT, |R16|, |R10|, PT;                      /* 0x1a0e00002901dcc1 */
        /*0538*/                   SEL R18, R20, R18, P0;                                      /* 0x2000000049449c04 */
                                                                                               /* 0x22d28042b2d2e047 */
        /*0548*/                   SEL R19, R21, R19, P0;                                      /* 0x200000004d54dc04 */
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
        /*05b0*/                   DMUL R18, R14, R16;                                         /* 0x5000000040e49c01 */
        /*05b8*/                   BRA 0x5d0;                                                  /* 0x4000000040001de7 */
                                                                                               /* 0x20000002e20042b7 */
        /*05c8*/                   DMUL R18, R14, R10;                                         /* 0x5000000028e49c01 */
        /*05d0*/                   MOV R22, R18;                                               /* 0x2800000048059de4 */
        /*05d8*/                   MOV R23, R19;                                               /* 0x280000004c05dde4 */
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
        /*00e0*/                   LD.E.64 R20, [R10+0xfa50];                                  /* 0x840003e940a51ca5 */
        /*00e8*/                   IADD R14.CC, R22, R2;                                       /* 0x4801000009639c03 */
        /*00f0*/               @P0 IADD R3, R4, R5;                                            /* 0x480000001440c003 */
        /*00f8*/                   LD.E.64 R4, [R10+0x1f490];                                  /* 0x840007d240a11ca5 */
                                                                                               /* 0x225233f230428047 */
        /*0108*/                   IADD.X R15, R0, R3;                                         /* 0x480000000c03dc43 */
        /*0110*/                   MOV32I R10, 0x0;                                            /* 0x1800000000029de2 */
        /*0118*/                   LD.E.64 R8, [R14+0x1f490];                                  /* 0x840007d240e21ca5 */
        /*0120*/                   MOV32I R11, 0x3fe00000;                                     /* 0x18ff80000002dde2 */
        /*0128*/                   LD.E.64 R6, [R14+0xfa50];                                   /* 0x840003e940e19ca5 */
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
        /*03c8*/                   LD.E.64 R12, [R8+0x4e350];                                  /* 0x8400138d40831ca5 */
        /*03d0*/                   DFMA R14, -R6, R18, c[0x2][0x0];                            /* 0x2024880000639e01 */
        /*03d8*/                   LD.E.64 R20, [R8+0x5dd90];                                  /* 0x8400177640851ca5 */
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
        /*0438*/                   ST.E.64 [R8+0x4e350], R2;                                   /* 0x9400138d40809ca5 */
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
        /*0498*/                   ST.E.64 [R8+0x5dd90], R10;                                  /* 0x9400177640829ca5 */
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
        /*0090*/                   ST.E.64 [R2+0x4e350], RZ;         /* 0x9400138d402fdca5 */
        /*0098*/                   ST.E.64 [R2+0x5dd90], RZ;         /* 0x94001776402fdca5 */
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


.global .align 16 .b8 __Bodydata_buffer[448464];
.global .align 8 .u64 _ZN4ikra8executor4cuda14d_storage_sizeE;
.global .align 8 .u64 _ZN4ikra8executor4cuda19d_storage_data_headE;
.global .align 8 .u64 _ZN4ikra3soa23d_previous_storage_sizeE;
.global .align 8 .u64 _ZN4ikra3soa18d_arena_allocationE;

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
st.global.u64 [%rd4+320336], %rd5;
st.global.u64 [%rd4+384400], %rd5;
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
add.s64 %rd7, %rd6, 64080;
add.s64 %rd8, %rd7, %rd5;
shl.b64 %rd9, %rd1, 3;
add.s64 %rd10, %rd7, %rd9;
ld.global.f64 %fd1, [%rd10];
ld.global.f64 %fd2, [%rd8];
sub.f64 %fd3, %fd2, %fd1;
add.s64 %rd11, %rd6, 128144;
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
ld.global.f64 %fd18, [%rd17+320336];
add.f64 %fd19, %fd17, %fd18;
ld.global.f64 %fd20, [%rd17+384400];
st.global.f64 [%rd17+320336], %fd19;
mul.f64 %fd21, %fd6, %fd15;
div.rn.f64 %fd22, %fd21, %fd9;
add.f64 %fd23, %fd22, %fd20;
st.global.f64 [%rd17+384400], %fd23;

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
ld.global.f64 %fd5, [%rd7+320336];
mul.f64 %fd6, %fd5, %fd4;
ld.global.f64 %fd7, [%rd7+16];
div.rn.f64 %fd8, %fd6, %fd7;
add.s64 %rd1, %rd7, 192208;
ld.global.f64 %fd9, [%rd7+192208];
ld.global.f64 %fd10, [%rd7+384400];
ld.global.f64 %fd11, [%rd7+256272];
add.f64 %fd12, %fd8, %fd9;
ld.global.f64 %fd13, [%rd7+64080];
ld.global.f64 %fd14, [%rd7+128144];
st.global.f64 [%rd7+192208], %fd12;
mul.f64 %fd15, %fd10, %fd4;
ld.global.f64 %fd16, [%rd7+16];
div.rn.f64 %fd17, %fd15, %fd16;
add.s64 %rd2, %rd7, 256272;
add.f64 %fd18, %fd17, %fd11;
st.global.f64 [%rd7+256272], %fd18;
fma.rn.f64 %fd19, %fd12, %fd4, %fd13;
st.global.f64 [%rd7+64080], %fd19;
ld.global.f64 %fd20, [%rd7+256272];
add.s64 %rd3, %rd7, 128144;
fma.rn.f64 %fd26, %fd20, %fd4, %fd14;
ld.global.f64 %fd21, [%rd7+64080];
st.global.f64 [%rd7+128144], %fd26;
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
st.global.u64 [%rd4+320336], %rd5;
st.global.u64 [%rd4+384400], %rd5;
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
st.global.u64 [%rd4+320336], %rd5;
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
add.s64 %rd1, %rd11, 320336;
add.s64 %rd2, %rd11, 16;
add.s64 %rd3, %rd11, 192208;
add.s64 %rd4, %rd11, 384400;
add.s64 %rd5, %rd11, 256272;
add.s64 %rd6, %rd11, 64080;
add.s64 %rd7, %rd11, 128144;
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
st.global.u64 [%rd5+320344], %rd6;
st.global.u64 [%rd5+384408], %rd6;

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
add.s64 %rd3, %rd28, 320344;
add.s64 %rd4, %rd28, 384408;
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
add.s64 %rd31, %rd30, 64080;
add.s64 %rd32, %rd31, %rd29;
mul.wide.s32 %rd33, %r4, 8;
add.s64 %rd34, %rd33, 8;
add.s64 %rd35, %rd31, %rd34;
ld.global.f64 %fd1, [%rd35];
ld.global.f64 %fd2, [%rd32];
sub.f64 %fd3, %fd2, %fd1;
add.s64 %rd36, %rd30, 128144;
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
add.s64 %rd42, %rd41, 64080;
add.s64 %rd43, %rd42, %rd40;
mul.wide.s32 %rd44, %r4, 8;
add.s64 %rd45, %rd44, 8;
add.s64 %rd46, %rd42, %rd45;
ld.global.f64 %fd24, [%rd46];
ld.global.f64 %fd25, [%rd43];
sub.f64 %fd26, %fd25, %fd24;
add.s64 %rd47, %rd41, 128144;
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
add.s64 %rd53, %rd52, 64080;
add.s64 %rd54, %rd53, %rd51;
mul.wide.s32 %rd55, %r4, 8;
add.s64 %rd56, %rd55, 8;
add.s64 %rd57, %rd53, %rd56;
ld.global.f64 %fd47, [%rd57];
ld.global.f64 %fd48, [%rd54];
sub.f64 %fd49, %fd48, %fd47;
add.s64 %rd58, %rd52, 128144;
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
add.s64 %rd64, %rd63, 64080;
add.s64 %rd65, %rd64, %rd62;
mul.wide.s32 %rd66, %r4, 8;
add.s64 %rd67, %rd66, 8;
add.s64 %rd68, %rd64, %rd67;
ld.global.f64 %fd70, [%rd68];
ld.global.f64 %fd71, [%rd65];
sub.f64 %fd72, %fd71, %fd70;
add.s64 %rd69, %rd63, 128144;
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
add.s64 %rd75, %rd74, 64080;
add.s64 %rd76, %rd75, %rd73;
mul.wide.s32 %rd77, %r4, 8;
add.s64 %rd78, %rd77, 8;
add.s64 %rd79, %rd75, %rd78;
ld.global.f64 %fd93, [%rd79];
ld.global.f64 %fd94, [%rd76];
sub.f64 %fd95, %fd94, %fd93;
add.s64 %rd80, %rd74, 128144;
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
ld.global.f64 %fd5, [%rd8+320344];
mul.f64 %fd6, %fd5, %fd4;
ld.global.f64 %fd7, [%rd8+24];
div.rn.f64 %fd8, %fd6, %fd7;
add.s64 %rd2, %rd8, 192216;
ld.global.f64 %fd9, [%rd8+192216];
ld.global.f64 %fd10, [%rd8+384408];
ld.global.f64 %fd11, [%rd8+256280];
add.f64 %fd12, %fd8, %fd9;
ld.global.f64 %fd13, [%rd8+64088];
ld.global.f64 %fd14, [%rd8+128152];
st.global.f64 [%rd8+192216], %fd12;
mul.f64 %fd15, %fd10, %fd4;
ld.global.f64 %fd16, [%rd8+24];
div.rn.f64 %fd17, %fd15, %fd16;
add.s64 %rd3, %rd8, 256280;
add.f64 %fd18, %fd17, %fd11;
st.global.f64 [%rd8+256280], %fd18;
fma.rn.f64 %fd19, %fd12, %fd4, %fd13;
st.global.f64 [%rd8+64088], %fd19;
ld.global.f64 %fd20, [%rd8+256280];
add.s64 %rd4, %rd8, 128152;
fma.rn.f64 %fd26, %fd20, %fd4, %fd14;
ld.global.f64 %fd21, [%rd8+64088];
st.global.f64 [%rd8+128152], %fd26;
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
add.s64 %rd2, %rd3, 320344;
add.s64 %rd4, %rd3, 192216;
add.s64 %rd5, %rd3, 384408;
add.s64 %rd6, %rd3, 256280;
add.s64 %rd7, %rd3, 64088;
add.s64 %rd8, %rd3, 128152;
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


.visible .entry _ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I4BodyLm0EEEEEvPT_m(
.param .u64 _ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I4BodyLm0EEEEEvPT_m_param_0,
.param .u64 _ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I4BodyLm0EEEEEvPT_m_param_1
)
{
.reg .b32 %r<3>;
.reg .b64 %rd<7>;


ld.param.u64 %rd1, [_ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I4BodyLm0EEEEEvPT_m_param_0];
cvta.to.global.u64 %rd2, %rd1;
ld.param.u32 %r1, [_ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I4BodyLm0EEEEEvPT_m_param_1];
add.s64 %rd3, %rd2, 8;
atom.global.add.u32 %r2, [%rd3], %r1;
cvt.u64.u32	%rd4, %r2;
add.s64 %rd5, %rd1, %rd4;
add.s64 %rd6, %rd5, 8;
st.global.u64 [_ZN4ikra3soa18d_arena_allocationE], %rd6;
ret;
}


.visible .entry _ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I4BodyLm0EEEEEvPT_m(
.param .u64 _ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I4BodyLm0EEEEEvPT_m_param_0,
.param .u64 _ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I4BodyLm0EEEEEvPT_m_param_1
)
{
.reg .b64 %rd<5>;


ld.param.u64 %rd1, [_ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I4BodyLm0EEEEEvPT_m_param_0];
ld.param.u64 %rd2, [_ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I4BodyLm0EEEEEvPT_m_param_1];
cvta.to.global.u64 %rd3, %rd1;
atom.global.add.u64 %rd4, [%rd3], %rd2;
st.global.u64 [_ZN4ikra3soa23d_previous_storage_sizeE], %rd4;
ret;
}


.visible .entry _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I4BodyLm0EEEJEEEvPT_DpT0_(
.param .u64 _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I4BodyLm0EEEJEEEvPT_DpT0__param_0
)
{
.reg .pred %p<2>;
.reg .b32 %r<2>;
.reg .b64 %rd<4>;


ld.param.u64 %rd1, [_ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I4BodyLm0EEEJEEEvPT_DpT0__param_0];
setp.eq.s64	%p1, %rd1, 0;
@%p1 bra BB13_2;

cvta.to.global.u64 %rd2, %rd1;
mov.u64 %rd3, 0;
st.global.u64 [%rd2], %rd3;
mov.u32 %r1, 0;
st.global.u32 [%rd2+8], %r1;

BB13_2:
ret;
}


