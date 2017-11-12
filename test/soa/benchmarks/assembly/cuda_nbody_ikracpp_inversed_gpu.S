
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
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ10run_simplevENKUlvE_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7_:
  REG:32 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:352 CONSTANT[2]:36 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE1_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7_:
  REG:32 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:352 CONSTANT[2]:36 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE0_clEvEUlPT_DpT0_E_JP4BodyEEEvS4_mS7_:
  REG:45 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:344 CONSTANT[2]:64 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE_clEvEUlPT_DpT0_E_JP4BodyEEEvS4_mS7_:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:344 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I4BodyLm0EEEJEEEvPT_DpT0_:
  REG:4 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra3soa28storage_increase_size_kernelEPNS0_19StorageStrategyBaseEm:
  REG:10 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z22codengen_simple_updateP4Bodyd:
  REG:32 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 CONSTANT[2]:36 TEXTURE:0 SURFACE:0 SAMPLER:0
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
		Function : _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ10run_simplevENKUlvE_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                               /* 0x22c2b202828232b7 */
        /*0008*/                   MOV R1, c[0x0][0x44];                                       /* 0x2800400110005de4 */
        /*0010*/                   S2R R0, SR_CTAID.X;                                         /* 0x2c00000094001c04 */
        /*0018*/                   S2R R3, SR_TID.X;                                           /* 0x2c0000008400dc04 */
        /*0020*/                   IMAD R0, R0, c[0x0][0x28], R3;                              /* 0x20064000a0001ca3 */
        /*0028*/                   ISUB RZ.CC, R0, c[0x0][0x140];                              /* 0x48014005000fdd03 */
        /*0030*/                   SHR R2, R0, 0x1f;                                           /* 0x5800c0007c009c23 */
        /*0038*/                   ISETP.GE.U32.X.AND P0, PT, R2, c[0x0][0x144], PT;           /* 0x1b0e40051021dc43 */
                                                                                               /* 0x2200420042c042e7 */
        /*0048*/               @P0 EXIT;                                                       /* 0x80000000000001e7 */
        /*0050*/                   IADD R4.CC, R0, c[0x0][0x150];                              /* 0x4801400540011c03 */
        /*0058*/                   MOV32I R0, 0x3;                                             /* 0x180000000c001de2 */
        /*0060*/                   IADD.X R2, R2, c[0x0][0x154];                               /* 0x4800400550209c43 */
        /*0068*/                   SHL R3, R4, -0x1d;                                          /* 0x6000ffff8c40dc03 */
        /*0070*/                   IADD32I RZ.CC, R0, -0x20;                                   /* 0x0fffffff800fdc02 */
        /*0078*/                   MOV32I R6, 0x1;                                             /* 0x1800000004019de2 */
                                                                                               /* 0x22b0420242421047 */
        /*0088*/                   MOV32I R0, 0x0;                                             /* 0x1800000000001de2 */
        /*0090*/                   SSY 0x198;                                                  /* 0x6000000400001c07 */
        /*0098*/                   SHR.U32 R5, R4, 0x1d;                                       /* 0x5800c00074415c03 */
        /*00a0*/                   SHL R2, R2, 0x3;                                            /* 0x6000c0000c209c03 */
        /*00a8*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*00b0*/                   IMAD.U32.U32 R4.CC, R4, 0x8, R0;                            /* 0x2001c00020411c03 */
        /*00b8*/                   MOV32I R0, 0x0;                                             /* 0x1800000000001de2 */
                                                                                               /* 0x229283f232804287 */
        /*00c8*/               @P0 IADD R3, R2, R5;                                            /* 0x480000001420c003 */
        /*00d0*/                   MOV R2, R4;                                                 /* 0x2800000010009de4 */
        /*00d8*/                   IADD.X R3, R0, R3;                                          /* 0x480000000c00dc43 */
        /*00e0*/                   LD.E.64 R8, [R2+0x10];                                      /* 0x8400000040221ca5 */
        /*00e8*/                   LD.E.64 R4, [R2+0x4e210];                                   /* 0x8400138840211ca5 */
        /*00f0*/                   MUFU.RCP64H R7, R9;                                         /* 0xc80000001891dc00 */
        /*00f8*/                   DFMA R10, -R8, R6, c[0x2][0x0];                             /* 0x200c880000829e01 */
                                                                                               /* 0x2282029292925237 */
        /*0108*/                   DFMA R10, R10, R10, R10;                                    /* 0x2014000028a29c01 */
        /*0110*/                   DMUL R14, R4, c[0x0][0x158];                                /* 0x5000400560439c01 */
        /*0118*/                   DFMA R10, R6, R10, R6;                                      /* 0x200c000028629c01 */
        /*0120*/                   DMUL R12, R14, R10;                                         /* 0x5000000028e31c01 */
        /*0128*/                   DFMA R6, -R8, R12, R14;                                     /* 0x201c000030819e01 */
        /*0130*/                   DFMA R12, R10, R6, R12;                                     /* 0x2018000018a31c01 */
        /*0138*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                 /* 0x160f480020ffdc80 */
                                                                                               /* 0x2202b2b042b2c287 */
        /*0148*/                   FFMA R0, RZ, R9, R13;                                       /* 0x301a000027f01c00 */
        /*0150*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0xc], PT;                 /* 0x220e48003001dc80 */
        /*0158*/               @P0 BRA CC.NEU, 0x190;                                          /* 0x40000000c00001a7 */
        /*0160*/                   MOV R10, R8;                                                /* 0x2800000020029de4 */
        /*0168*/                   MOV R11, R9;                                                /* 0x280000002402dde4 */
        /*0170*/                   CAL 0x2c0;                                                  /* 0x5000000520011c07 */
        /*0178*/                   MOV R12, R22;                                               /* 0x2800000058031de4 */
                                                                                               /* 0x223230423042b2b7 */
        /*0188*/                   MOV R13, R23;                                               /* 0x280000005c035de4 */
        /*0190*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0198*/                   LD.E.64 R10, [R2+0x10];                                     /* 0x8400000040229ca5 */
        /*01a0*/                   MOV32I R20, 0x1;                                            /* 0x1800000004051de2 */
        /*01a8*/                   LD.E.64 R16, [R2+0x5dc10];                                  /* 0x8400177040241ca5 */
        /*01b0*/                   SSY 0x280;                                                  /* 0x6000000320001c07 */
        /*01b8*/                   LD.E.64 R14, [R2+0x2ee10];                                  /* 0x84000bb840239ca5 */
                                                                                               /* 0x2252329283f23237 */
        /*01c8*/                   LD.E.64 R4, [R2+0x3e810];                                   /* 0x84000fa040211ca5 */
        /*01d0*/                   LD.E.64 R6, [R2+0xfa10];                                    /* 0x840003e840219ca5 */
        /*01d8*/                   LD.E.64 R8, [R2+0x1f410];                                   /* 0x840007d040221ca5 */
        /*01e0*/                   MUFU.RCP64H R21, R11;                                       /* 0xc800000018b55c00 */
        /*01e8*/                   DFMA R18, -R10, R20, c[0x2][0x0];                           /* 0x2028880000a49e01 */
        /*01f0*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*01f8*/                   DMUL R16, R16, c[0x0][0x158];                               /* 0x5000400561041c01 */
                                                                                               /* 0x2200429232929297 */
        /*0208*/                   DFMA R18, R20, R18, R20;                                    /* 0x2028000049449c01 */
        /*0210*/                   DMUL R20, R16, R18;                                         /* 0x5000000049051c01 */
        /*0218*/                   DFMA R22, -R10, R20, R16;                                   /* 0x2020000050a59e01 */
        /*0220*/                   DFMA R22, R18, R22, R20;                                    /* 0x2028000059259c01 */
        /*0228*/                   DADD R12, R12, R14;                                         /* 0x4800000038c31c01 */
        /*0230*/                   FFMA R0, RZ, R11, R23;                                      /* 0x302e00002ff01c00 */
        /*0238*/                   ST.E.64 [R2+0x2ee10], R12;                                  /* 0x94000bb840231ca5 */
                                                                                               /* 0x22b2b2b042b2c277 */
        /*0248*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                 /* 0x160f4800211fdc80 */
        /*0250*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0xc], PT;                 /* 0x220e48003001dc80 */
        /*0258*/               @P0 BRA CC.NEU, 0x278;                                          /* 0x40000000600001a7 */
        /*0260*/                   MOV R14, R16;                                               /* 0x2800000040039de4 */
        /*0268*/                   MOV R15, R17;                                               /* 0x280000004403dde4 */
        /*0270*/                   CAL 0x2c0;                                                  /* 0x5000000120011c07 */
        /*0278*/                   DADD.S R4, R4, R22;                                         /* 0x4800000058411c11 */
                                                                                               /* 0x22e04293f2e2e047 */
        /*0288*/                   ST.E.64 [R2+0x3e810], R4;                                   /* 0x94000fa040211ca5 */
        /*0290*/                   DFMA R6, R12, c[0x0][0x158], R6;                            /* 0x200c400560c19c01 */
        /*0298*/                   LD.E.64 R10, [R2+0x3e810];                                  /* 0x84000fa040229ca5 */
        /*02a0*/                   ST.E.64 [R2+0xfa10], R6;                                    /* 0x940003e840219ca5 */
        /*02a8*/                   DFMA R8, R10, c[0x0][0x158], R8;                            /* 0x2010400560a21c01 */
        /*02b0*/                   ST.E.64 [R2+0x1f410], R8;                                   /* 0x940007d040221ca5 */
        /*02b8*/                   EXIT;                                                       /* 0x8000000000001de7 */
                                                                                               /* 0x2282b04200428047 */
        /*02c8*/                   LOP32I.AND R0, R11, 0x40000000;                             /* 0x3900000000b01c02 */
        /*02d0*/                   LOP32I.AND R26, R15, 0x7f800000;                            /* 0x39fe000000f69c02 */
        /*02d8*/                   ISETP.LT.U32.AND P0, PT, R0, c[0x2][0x10], PT;              /* 0x188e48004001dc03 */
        /*02e0*/                   MOV32I R0, 0x1ff00000;                                      /* 0x187fc00000001de2 */
        /*02e8*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*02f0*/                   MOV32I R20, 0x1;                                            /* 0x1800000004051de2 */
        /*02f8*/                   SEL R17, R0, c[0x2][0x14], !P0;                             /* 0x2010480050045c04 */
                                                                                               /* 0x2232928042c04207 */
        /*0308*/                   DMUL R18, R10, R16;                                         /* 0x5000000040a49c01 */
        /*0310*/                   ISETP.LT.U32.AND P0, PT, R26, c[0x2][0x18], PT;             /* 0x188e480061a1dc03 */
        /*0318*/                   MOV R24, RZ;                                                /* 0x28000000fc061de4 */
        /*0320*/                   MUFU.RCP64H R21, R19;                                       /* 0xc800000019355c00 */
        /*0328*/                   SEL R25, R0, c[0x2][0x14], !P0;                             /* 0x2010480050065c04 */
        /*0330*/                   DFMA R22, -R18, R20, c[0x2][0x0];                           /* 0x2028880001259e01 */
        /*0338*/                   DFMA R30, R22, R22, R22;                                    /* 0x202c000059679c01 */
                                                                                               /* 0x22b2d29292929257 */
        /*0348*/                   DMUL R22, R14, R24;                                         /* 0x5000000060e59c01 */
        /*0350*/                   DFMA R20, R20, R30, R20;                                    /* 0x2028000079451c01 */
        /*0358*/                   DMUL R24, R22, R20;                                         /* 0x5000000051661c01 */
        /*0360*/                   DFMA R18, -R18, R24, R22;                                   /* 0x202c000061249e01 */
        /*0368*/                   DFMA R22, R20, R18, R24;                                    /* 0x2030000049459c01 */
        /*0370*/                   DSETP.LEU.AND P0, PT, |R22|, RZ, PT;                        /* 0x1d8e0000fd61dc81 */
        /*0378*/               @P0 BRA 0x530;                                                  /* 0x40000006c00001e7 */
                                                                                               /* 0x2232523282823047 */
        /*0388*/                   ISETP.GT.U32.AND P0, PT, R26, c[0x2][0x1c], PT;             /* 0x1a0e480071a1dc03 */
        /*0390*/                   MOV R18, RZ;                                                /* 0x28000000fc049de4 */
        /*0398*/                   DMUL R20, R16, R22;                                         /* 0x5000000059051c01 */
        /*03a0*/                   SEL R19, R0, c[0x2][0x14], !P0;                             /* 0x201048005004dc04 */
        /*03a8*/                   DMUL R22, R22, R18;                                         /* 0x5000000049659c01 */
        /*03b0*/                   DMUL R20, R18, R20;                                         /* 0x5000000051251c01 */
        /*03b8*/                   DMUL R16, R16, R22;                                         /* 0x5000000059041c01 */
                                                                                               /* 0x22b2c27202d29257 */
        /*03c8*/                   DFMA R18, R10, R20, -R14;                                   /* 0x201c000050a49d01 */
        /*03d0*/                   DFMA R22, R10, R16, -R14;                                   /* 0x201c000040a59d01 */
        /*03d8*/                   DSETP.GT.AND P0, PT, |R18|, |R22|, PT;                      /* 0x1a0e00005921dcc1 */
        /*03e0*/                   SEL R17, R17, R21, P0;                                      /* 0x2000000055145c04 */
        /*03e8*/                   SEL R16, R16, R20, P0;                                      /* 0x2000000051041c04 */
        /*03f0*/                   FSETP.GTU.AND P0, PT, |R17|, 1.469367938527859385e-39, PT;  /* 0x260ec0040111dc80 */
        /*03f8*/               @P0 BRA 0x5a8;                                                  /* 0x40000006a00001e7 */
                                                                                               /* 0x228042a200420047 */
        /*0408*/                   FSETP.LT.AND P0, PT, |R15|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000f1dc80 */
        /*0410*/                   LOP32I.AND R0, R16, 0xfffffffe;                             /* 0x3bfffffff9001c02 */
        /*0418*/                   MOV32I R21, 0x3ff00000;                                     /* 0x18ffc00000055de2 */
        /*0420*/                   MOV R20, RZ;                                                /* 0x28000000fc051de4 */
        /*0428*/                   LOP32I.OR R22, R16, 0x1;                                    /* 0x3800000005059c42 */
        /*0430*/                   SEL R21, R21, c[0x2][0x20], !P0;                            /* 0x2010480081555c04 */
        /*0438*/                   MOV R16, R0;                                                /* 0x2800000000041de4 */
                                                                                               /* 0x22d2923280420237 */
        /*0448*/                   DMUL R10, R10, R20;                                         /* 0x5000000050a29c01 */
        /*0450*/                   DMUL R14, R14, R20;                                         /* 0x5000000050e39c01 */
        /*0458*/                   MOV R20, R22;                                               /* 0x2800000058051de4 */
        /*0460*/                   MOV R21, R17;                                               /* 0x2800000044055de4 */
        /*0468*/                   DFMA R18, R16, R10, -R14;                                   /* 0x201c000029049d01 */
        /*0470*/                   DFMA R20, R20, R10, -R14;                                   /* 0x201c000029451d01 */
        /*0478*/                   DSETP.GT.AND P0, PT, |R18|, |R20|, PT;                      /* 0x1a0e00005121dcc1 */
                                                                                               /* 0x2202b20042c04287 */
        /*0488*/                   SEL R23, R22, R0, P0;                                       /* 0x200000000165dc04 */
        /*0490*/                   IADD32I R0.CC, R23, 0x1;                                    /* 0x0c00000005701c02 */
        /*0498*/                   LOP32I.AND R18, R23, 0x1;                                   /* 0x3800000005749c02 */
        /*04a0*/                   ISETP.EQ.U32.AND P0, PT, R18, 0x1, PT;                      /* 0x190ec0000521dc03 */
        /*04a8*/                   IADD.X R16, RZ, R17;                                        /* 0x4800000047f41c43 */
        /*04b0*/                   IADD32I R20.CC, R23, -0x1;                                  /* 0x0ffffffffd751c02 */
        /*04b8*/                   SEL R0, R23, R0, !P0;                                       /* 0x2010000001701c04 */
                                                                                               /* 0x2232820042720047 */
        /*04c8*/                   IADD32I.X R21, R17, -0x1;                                   /* 0x0bfffffffd155c42 */
        /*04d0*/                   SEL R19, R17, R16, !P0;                                     /* 0x201000004114dc04 */
        /*04d8*/                   SEL R20, R20, R23, !P0;                                     /* 0x201000005d451c04 */
        /*04e0*/                   MOV R16, R0;                                                /* 0x2800000000041de4 */
        /*04e8*/                   SEL R21, R21, R17, !P0;                                     /* 0x2010000045555c04 */
        /*04f0*/                   MOV R17, R19;                                               /* 0x280000004c045de4 */
        /*04f8*/                   DFMA R16, R10, R16, -R14;                                   /* 0x201c000040a41d01 */
                                                                                               /* 0x22b2d2e04202d297 */
        /*0508*/                   DFMA R10, R10, R20, -R14;                                   /* 0x201c000050a29d01 */
        /*0510*/                   DSETP.GT.AND P0, PT, |R16|, |R10|, PT;                      /* 0x1a0e00002901dcc1 */
        /*0518*/                   SEL R16, R20, R0, P0;                                       /* 0x2000000001441c04 */
        /*0520*/                   SEL R17, R21, R19, P0;                                      /* 0x200000004d545c04 */
        /*0528*/                   BRA 0x5a8;                                                  /* 0x40000001e0001de7 */
        /*0530*/                   DSETP.EQ.AND P0, PT, R22, RZ, PT;                           /* 0x190e0000fd61dc01 */
        /*0538*/               @P0 BRA 0x5a0;                                                  /* 0x40000001800001e7 */
                                                                                               /* 0x22027202d2d28047 */
        /*0548*/                   MUFU.RCP64H R17, R11;                                       /* 0xc800000018b45c00 */
        /*0550*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*0558*/                   DSETP.GT.AND P1, PT, |R16|, RZ, PT;                         /* 0x1a0e0000fd03dc81 */
        /*0560*/              @!P1 DSETP.NEU.AND P0, PT, |R10|, +INF , PT;                     /* 0x1e8edffc00a1e481 */
        /*0568*/              @!P1 SEL R0, R10, R16, P0;                                       /* 0x2000000040a02404 */
        /*0570*/              @!P1 SEL R11, R11, R17, P0;                                      /* 0x2000000044b2e404 */
        /*0578*/              @!P1 MOV R16, R0;                                                /* 0x28000000000425e4 */
                                                                                               /* 0x22e20042b2e20287 */
        /*0588*/              @!P1 MOV R17, R11;                                               /* 0x280000002c0465e4 */
        /*0590*/                   DMUL R16, R14, R16;                                         /* 0x5000000040e41c01 */
        /*0598*/                   BRA 0x5a8;                                                  /* 0x4000000020001de7 */
        /*05a0*/                   DMUL R16, R14, R10;                                         /* 0x5000000028e41c01 */
        /*05a8*/                   MOV R22, R16;                                               /* 0x2800000040059de4 */
        /*05b0*/                   MOV R23, R17;                                               /* 0x280000004405dde4 */
        /*05b8*/                   RET;                                                        /* 0x9000000000001de7 */
        /*05c0*/                   BRA 0x5c0;                                                  /* 0x4003ffffe0001de7 */
        /*05c8*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*05d0*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*05d8*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*05e0*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*05e8*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*05f0*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*05f8*/                   NOP;                                                        /* 0x4000000000001de4 */
		.................................................................................................................


		Function : _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE1_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                               /* 0x22c2b202828232b7 */
        /*0008*/                   MOV R1, c[0x0][0x44];                                       /* 0x2800400110005de4 */
        /*0010*/                   S2R R0, SR_CTAID.X;                                         /* 0x2c00000094001c04 */
        /*0018*/                   S2R R3, SR_TID.X;                                           /* 0x2c0000008400dc04 */
        /*0020*/                   IMAD R0, R0, c[0x0][0x28], R3;                              /* 0x20064000a0001ca3 */
        /*0028*/                   ISUB RZ.CC, R0, c[0x0][0x140];                              /* 0x48014005000fdd03 */
        /*0030*/                   SHR R2, R0, 0x1f;                                           /* 0x5800c0007c009c23 */
        /*0038*/                   ISETP.GE.U32.X.AND P0, PT, R2, c[0x0][0x144], PT;           /* 0x1b0e40051021dc43 */
                                                                                               /* 0x2200420042c042e7 */
        /*0048*/               @P0 EXIT;                                                       /* 0x80000000000001e7 */
        /*0050*/                   IADD R4.CC, R0, c[0x0][0x150];                              /* 0x4801400540011c03 */
        /*0058*/                   MOV32I R0, 0x3;                                             /* 0x180000000c001de2 */
        /*0060*/                   IADD.X R2, R2, c[0x0][0x154];                               /* 0x4800400550209c43 */
        /*0068*/                   SHL R3, R4, -0x1d;                                          /* 0x6000ffff8c40dc03 */
        /*0070*/                   IADD32I RZ.CC, R0, -0x20;                                   /* 0x0fffffff800fdc02 */
        /*0078*/                   MOV32I R6, 0x1;                                             /* 0x1800000004019de2 */
                                                                                               /* 0x22b0420242421047 */
        /*0088*/                   MOV32I R0, 0x0;                                             /* 0x1800000000001de2 */
        /*0090*/                   SSY 0x198;                                                  /* 0x6000000400001c07 */
        /*0098*/                   SHR.U32 R5, R4, 0x1d;                                       /* 0x5800c00074415c03 */
        /*00a0*/                   SHL R2, R2, 0x3;                                            /* 0x6000c0000c209c03 */
        /*00a8*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*00b0*/                   IMAD.U32.U32 R4.CC, R4, 0x8, R0;                            /* 0x2001c00020411c03 */
        /*00b8*/                   MOV32I R0, 0x0;                                             /* 0x1800000000001de2 */
                                                                                               /* 0x229283f232804287 */
        /*00c8*/               @P0 IADD R3, R2, R5;                                            /* 0x480000001420c003 */
        /*00d0*/                   MOV R2, R4;                                                 /* 0x2800000010009de4 */
        /*00d8*/                   IADD.X R3, R0, R3;                                          /* 0x480000000c00dc43 */
        /*00e0*/                   LD.E.64 R8, [R2+0x10];                                      /* 0x8400000040221ca5 */
        /*00e8*/                   LD.E.64 R4, [R2+0x4e210];                                   /* 0x8400138840211ca5 */
        /*00f0*/                   MUFU.RCP64H R7, R9;                                         /* 0xc80000001891dc00 */
        /*00f8*/                   DFMA R10, -R8, R6, c[0x2][0x0];                             /* 0x200c880000829e01 */
                                                                                               /* 0x2282029292925237 */
        /*0108*/                   DFMA R10, R10, R10, R10;                                    /* 0x2014000028a29c01 */
        /*0110*/                   DMUL R14, R4, c[0x0][0x158];                                /* 0x5000400560439c01 */
        /*0118*/                   DFMA R10, R6, R10, R6;                                      /* 0x200c000028629c01 */
        /*0120*/                   DMUL R12, R14, R10;                                         /* 0x5000000028e31c01 */
        /*0128*/                   DFMA R6, -R8, R12, R14;                                     /* 0x201c000030819e01 */
        /*0130*/                   DFMA R12, R10, R6, R12;                                     /* 0x2018000018a31c01 */
        /*0138*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0x8], PT;                 /* 0x160f480020ffdc80 */
                                                                                               /* 0x2202b2b042b2c287 */
        /*0148*/                   FFMA R0, RZ, R9, R13;                                       /* 0x301a000027f01c00 */
        /*0150*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0xc], PT;                 /* 0x220e48003001dc80 */
        /*0158*/               @P0 BRA CC.NEU, 0x190;                                          /* 0x40000000c00001a7 */
        /*0160*/                   MOV R10, R8;                                                /* 0x2800000020029de4 */
        /*0168*/                   MOV R11, R9;                                                /* 0x280000002402dde4 */
        /*0170*/                   CAL 0x350;                                                  /* 0x5000000760011c07 */
        /*0178*/                   MOV R12, R22;                                               /* 0x2800000058031de4 */
                                                                                               /* 0x223230423042b2b7 */
        /*0188*/                   MOV R13, R23;                                               /* 0x280000005c035de4 */
        /*0190*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0198*/                   LD.E.64 R10, [R2+0x10];                                     /* 0x8400000040229ca5 */
        /*01a0*/                   MOV32I R20, 0x1;                                            /* 0x1800000004051de2 */
        /*01a8*/                   LD.E.64 R16, [R2+0x5dc10];                                  /* 0x8400177040241ca5 */
        /*01b0*/                   SSY 0x280;                                                  /* 0x6000000320001c07 */
        /*01b8*/                   LD.E.64 R14, [R2+0x2ee10];                                  /* 0x84000bb840239ca5 */
                                                                                               /* 0x2252329283f23237 */
        /*01c8*/                   LD.E.64 R4, [R2+0x3e810];                                   /* 0x84000fa040211ca5 */
        /*01d0*/                   LD.E.64 R6, [R2+0xfa10];                                    /* 0x840003e840219ca5 */
        /*01d8*/                   LD.E.64 R8, [R2+0x1f410];                                   /* 0x840007d040221ca5 */
        /*01e0*/                   MUFU.RCP64H R21, R11;                                       /* 0xc800000018b55c00 */
        /*01e8*/                   DFMA R18, -R10, R20, c[0x2][0x0];                           /* 0x2028880000a49e01 */
        /*01f0*/                   DFMA R18, R18, R18, R18;                                    /* 0x2024000049249c01 */
        /*01f8*/                   DMUL R16, R16, c[0x0][0x158];                               /* 0x5000400561041c01 */
                                                                                               /* 0x2200429232929297 */
        /*0208*/                   DFMA R18, R20, R18, R20;                                    /* 0x2028000049449c01 */
        /*0210*/                   DMUL R20, R16, R18;                                         /* 0x5000000049051c01 */
        /*0218*/                   DFMA R22, -R10, R20, R16;                                   /* 0x2020000050a59e01 */
        /*0220*/                   DFMA R22, R18, R22, R20;                                    /* 0x2028000059259c01 */
        /*0228*/                   DADD R12, R12, R14;                                         /* 0x4800000038c31c01 */
        /*0230*/                   FFMA R0, RZ, R11, R23;                                      /* 0x302e00002ff01c00 */
        /*0238*/                   ST.E.64 [R2+0x2ee10], R12;                                  /* 0x94000bb840231ca5 */
                                                                                               /* 0x22b2b2b042b2c277 */
        /*0248*/                   FSET.GTU.AND RZ.CC, |R17|, c[0x2][0x8], PT;                 /* 0x160f4800211fdc80 */
        /*0250*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0xc], PT;                 /* 0x220e48003001dc80 */
        /*0258*/               @P0 BRA CC.NEU, 0x278;                                          /* 0x40000000600001a7 */
        /*0260*/                   MOV R14, R16;                                               /* 0x2800000040039de4 */
        /*0268*/                   MOV R15, R17;                                               /* 0x280000004403dde4 */
        /*0270*/                   CAL 0x350;                                                  /* 0x5000000360011c07 */
        /*0278*/                   DADD.S R4, R4, R22;                                         /* 0x4800000058411c11 */
                                                                                               /* 0x2233f232e2e04207 */
        /*0288*/                   DFMA R6, R12, c[0x0][0x158], R6;                            /* 0x200c400560c19c01 */
        /*0290*/                   ST.E.64 [R2+0x3e810], R4;                                   /* 0x94000fa040211ca5 */
        /*0298*/                   SSY 0x310;                                                  /* 0x60000001c0001c07 */
        /*02a0*/                   ST.E.64 [R2+0xfa10], R6;                                    /* 0x940003e840219ca5 */
        /*02a8*/                   LD.E.64 R12, [R2+0xfa10];                                   /* 0x840003e840231ca5 */
        /*02b0*/                   LD.E.64 R10, [R2+0x3e810];                                  /* 0x84000fa040229ca5 */
        /*02b8*/                   DSETP.GT.AND P0, PT, R12, 1, PT;                            /* 0x1a0ecffc00c1dc01 */
                                                                                               /* 0x2283f232b2d04297 */
        /*02c8*/                   DFMA R8, R10, c[0x0][0x158], R8;                            /* 0x2010400560a21c01 */
        /*02d0*/                   ST.E.64 [R2+0x1f410], R8;                                   /* 0x940007d040221ca5 */
        /*02d8*/                   DSETP.LT.OR P0, PT, R12, -1, P0;                            /* 0x18a0effc00c1dc01 */
        /*02e0*/              @!P0 NOP.S;                                                      /* 0x40000000000021f4 */
        /*02e8*/                   LD.E.64 R4, [R2+0x2ee10];                                   /* 0x84000bb840211ca5 */
        /*02f0*/                   LD.E.64 R8, [R2+0x1f410];                                   /* 0x840007d040221ca5 */
        /*02f8*/                   F2F.F64.F64 R4, -R4;                                        /* 0x1000000011b11d04 */
                                                                                               /* 0x220283f2e2d2d2b7 */
        /*0308*/                   ST.E.64.S [R2+0x2ee10], R4;                                 /* 0x94000bb840211cb5 */
        /*0310*/                   DSETP.GT.AND P0, PT, R8, 1, PT;                             /* 0x1a0ecffc0081dc01 */
        /*0318*/                   DSETP.LT.OR P0, PT, R8, -1, P0;                             /* 0x18a0effc0081dc01 */
        /*0320*/              @!P0 EXIT;                                                       /* 0x80000000000021e7 */
        /*0328*/                   LD.E.64 R4, [R2+0x3e810];                                   /* 0x84000fa040211ca5 */
        /*0330*/                   F2F.F64.F64 R4, -R4;                                        /* 0x1000000011b11d04 */
        /*0338*/                   ST.E.64 [R2+0x3e810], R4;                                   /* 0x94000fa040211ca5 */
                                                                                               /* 0x22b04200428042e7 */
        /*0348*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*0350*/                   LOP32I.AND R0, R11, 0x40000000;                             /* 0x3900000000b01c02 */
        /*0358*/                   LOP32I.AND R26, R15, 0x7f800000;                            /* 0x39fe000000f69c02 */
        /*0360*/                   ISETP.LT.U32.AND P0, PT, R0, c[0x2][0x10], PT;              /* 0x188e48004001dc03 */
        /*0368*/                   MOV32I R0, 0x1ff00000;                                      /* 0x187fc00000001de2 */
        /*0370*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*0378*/                   MOV32I R20, 0x1;                                            /* 0x1800000004051de2 */
                                                                                               /* 0x22928042c2028047 */
        /*0388*/                   SEL R17, R0, c[0x2][0x14], !P0;                             /* 0x2010480050045c04 */
        /*0390*/                   MOV R24, RZ;                                                /* 0x28000000fc061de4 */
        /*0398*/                   DMUL R18, R10, R16;                                         /* 0x5000000040a49c01 */
        /*03a0*/                   ISETP.LT.U32.AND P0, PT, R26, c[0x2][0x18], PT;             /* 0x188e480061a1dc03 */
        /*03a8*/                   MUFU.RCP64H R21, R19;                                       /* 0xc800000019355c00 */
        /*03b0*/                   SEL R25, R0, c[0x2][0x14], !P0;                             /* 0x2010480050065c04 */
        /*03b8*/                   DFMA R22, -R18, R20, c[0x2][0x0];                           /* 0x2028880001259e01 */
                                                                                               /* 0x22d2929292925237 */
        /*03c8*/                   DFMA R30, R22, R22, R22;                                    /* 0x202c000059679c01 */
        /*03d0*/                   DMUL R22, R14, R24;                                         /* 0x5000000060e59c01 */
        /*03d8*/                   DFMA R20, R20, R30, R20;                                    /* 0x2028000079451c01 */
        /*03e0*/                   DMUL R24, R22, R20;                                         /* 0x5000000051661c01 */
        /*03e8*/                   DFMA R18, -R18, R24, R22;                                   /* 0x202c000061249e01 */
        /*03f0*/                   DFMA R22, R20, R18, R24;                                    /* 0x2030000049459c01 */
        /*03f8*/                   DSETP.LEU.AND P0, PT, |R22|, RZ, PT;                        /* 0x1d8e0000fd61dc81 */
                                                                                               /* 0x22523282823042b7 */
        /*0408*/               @P0 BRA 0x5b8;                                                  /* 0x40000006a00001e7 */
        /*0410*/                   ISETP.GT.U32.AND P0, PT, R26, c[0x2][0x1c], PT;             /* 0x1a0e480071a1dc03 */
        /*0418*/                   MOV R18, RZ;                                                /* 0x28000000fc049de4 */
        /*0420*/                   DMUL R20, R16, R22;                                         /* 0x5000000059051c01 */
        /*0428*/                   SEL R19, R0, c[0x2][0x14], !P0;                             /* 0x201048005004dc04 */
        /*0430*/                   DMUL R22, R22, R18;                                         /* 0x5000000049659c01 */
        /*0438*/                   DMUL R20, R18, R20;                                         /* 0x5000000051251c01 */
                                                                                               /* 0x22c27202d2925237 */
        /*0448*/                   DMUL R16, R16, R22;                                         /* 0x5000000059041c01 */
        /*0450*/                   DFMA R18, R10, R20, -R14;                                   /* 0x201c000050a49d01 */
        /*0458*/                   DFMA R22, R10, R16, -R14;                                   /* 0x201c000040a59d01 */
        /*0460*/                   DSETP.GT.AND P0, PT, |R18|, |R22|, PT;                      /* 0x1a0e00005921dcc1 */
        /*0468*/                   SEL R17, R17, R21, P0;                                      /* 0x2000000055145c04 */
        /*0470*/                   SEL R16, R16, R20, P0;                                      /* 0x2000000051041c04 */
        /*0478*/                   FSETP.GTU.AND P0, PT, |R17|, 1.469367938527859385e-39, PT;  /* 0x260ec0040111dc80 */
                                                                                               /* 0x2282a200420042b7 */
        /*0488*/               @P0 BRA 0x630;                                                  /* 0x40000006800001e7 */
        /*0490*/                   FSETP.LT.AND P0, PT, |R15|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000f1dc80 */
        /*0498*/                   LOP32I.AND R0, R16, 0xfffffffe;                             /* 0x3bfffffff9001c02 */
        /*04a0*/                   MOV32I R21, 0x3ff00000;                                     /* 0x18ffc00000055de2 */
        /*04a8*/                   MOV R20, RZ;                                                /* 0x28000000fc051de4 */
        /*04b0*/                   LOP32I.OR R22, R16, 0x1;                                    /* 0x3800000005059c42 */
        /*04b8*/                   SEL R21, R21, c[0x2][0x20], !P0;                            /* 0x2010480081555c04 */
                                                                                               /* 0x2292328042023047 */
        /*04c8*/                   MOV R16, R0;                                                /* 0x2800000000041de4 */
        /*04d0*/                   DMUL R14, R14, R20;                                         /* 0x5000000050e39c01 */
        /*04d8*/                   DMUL R10, R10, R20;                                         /* 0x5000000050a29c01 */
        /*04e0*/                   MOV R20, R22;                                               /* 0x2800000058051de4 */
        /*04e8*/                   MOV R21, R17;                                               /* 0x2800000044055de4 */
        /*04f0*/                   DFMA R18, R16, R10, -R14;                                   /* 0x201c000029049d01 */
        /*04f8*/                   DFMA R20, R20, R10, -R14;                                   /* 0x201c000029451d01 */
                                                                                               /* 0x22c20042c04282d7 */
        /*0508*/                   DSETP.GT.AND P0, PT, |R18|, |R20|, PT;                      /* 0x1a0e00005121dcc1 */
        /*0510*/                   SEL R23, R22, R0, P0;                                       /* 0x200000000165dc04 */
        /*0518*/                   IADD32I R0.CC, R23, 0x1;                                    /* 0x0c00000005701c02 */
        /*0520*/                   LOP32I.AND R18, R23, 0x1;                                   /* 0x3800000005749c02 */
        /*0528*/                   ISETP.EQ.U32.AND P0, PT, R18, 0x1, PT;                      /* 0x190ec0000521dc03 */
        /*0530*/                   IADD.X R16, RZ, R17;                                        /* 0x4800000047f41c43 */
        /*0538*/                   IADD32I R20.CC, R23, -0x1;                                  /* 0x0ffffffffd751c02 */
                                                                                               /* 0x2282020262020047 */
        /*0548*/                   SEL R0, R23, R0, !P0;                                       /* 0x2010000001701c04 */
        /*0550*/                   IADD32I.X R21, R17, -0x1;                                   /* 0x0bfffffffd155c42 */
        /*0558*/                   SEL R19, R17, R16, !P0;                                     /* 0x201000004114dc04 */
        /*0560*/                   SEL R20, R20, R23, !P0;                                     /* 0x201000005d451c04 */
        /*0568*/                   SEL R21, R21, R17, !P0;                                     /* 0x2010000045555c04 */
        /*0570*/                   MOV R16, R0;                                                /* 0x2800000000041de4 */
        /*0578*/                   MOV R17, R19;                                               /* 0x280000004c045de4 */
                                                                                               /* 0x22d2e04202d29237 */
        /*0588*/                   DFMA R16, R10, R16, -R14;                                   /* 0x201c000040a41d01 */
        /*0590*/                   DFMA R10, R10, R20, -R14;                                   /* 0x201c000050a29d01 */
        /*0598*/                   DSETP.GT.AND P0, PT, |R16|, |R10|, PT;                      /* 0x1a0e00002901dcc1 */
        /*05a0*/                   SEL R16, R20, R0, P0;                                       /* 0x2000000001441c04 */
        /*05a8*/                   SEL R17, R21, R19, P0;                                      /* 0x200000004d545c04 */
        /*05b0*/                   BRA 0x630;                                                  /* 0x40000001e0001de7 */
        /*05b8*/                   DSETP.EQ.AND P0, PT, R22, RZ, PT;                           /* 0x190e0000fd61dc01 */
                                                                                               /* 0x227202d2d28042b7 */
        /*05c8*/               @P0 BRA 0x628;                                                  /* 0x40000001600001e7 */
        /*05d0*/                   MUFU.RCP64H R17, R11;                                       /* 0xc800000018b45c00 */
        /*05d8*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*05e0*/                   DSETP.GT.AND P1, PT, |R16|, RZ, PT;                         /* 0x1a0e0000fd03dc81 */
        /*05e8*/              @!P1 DSETP.NEU.AND P0, PT, |R10|, +INF , PT;                     /* 0x1e8edffc00a1e481 */
        /*05f0*/              @!P1 SEL R0, R10, R16, P0;                                       /* 0x2000000040a02404 */
        /*05f8*/              @!P1 SEL R11, R11, R17, P0;                                      /* 0x2000000044b2e404 */
                                                                                               /* 0x220042b2e2028207 */
        /*0608*/              @!P1 MOV R16, R0;                                                /* 0x28000000000425e4 */
        /*0610*/              @!P1 MOV R17, R11;                                               /* 0x280000002c0465e4 */
        /*0618*/                   DMUL R16, R14, R16;                                         /* 0x5000000040e41c01 */
        /*0620*/                   BRA 0x630;                                                  /* 0x4000000020001de7 */
        /*0628*/                   DMUL R16, R14, R10;                                         /* 0x5000000028e41c01 */
        /*0630*/                   MOV R22, R16;                                               /* 0x2800000040059de4 */
        /*0638*/                   MOV R23, R17;                                               /* 0x280000004405dde4 */
                                                                                               /* 0x20000000000002e7 */
        /*0648*/                   RET;                                                        /* 0x9000000000001de7 */
        /*0650*/                   BRA 0x650;                                                  /* 0x4003ffffe0001de7 */
        /*0658*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*0660*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*0668*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*0670*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*0678*/                   NOP;                                                        /* 0x4000000000001de4 */
		......................................................................................................................


		Function : _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE0_clEvEUlPT_DpT0_E_JP4BodyEEEvS4_mS7_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                               /* 0x22c2b202828232b7 */
        /*0008*/                   MOV R1, c[0x0][0x44];                                       /* 0x2800400110005de4 */
        /*0010*/                   S2R R18, SR_CTAID.X;                                        /* 0x2c00000094049c04 */
        /*0018*/                   S2R R3, SR_TID.X;                                           /* 0x2c0000008400dc04 */
        /*0020*/                   IMAD R18, R18, c[0x0][0x28], R3;                            /* 0x20064000a1249ca3 */
        /*0028*/                   ISUB RZ.CC, R18, c[0x0][0x140];                             /* 0x48014005012fdd03 */
        /*0030*/                   SHR R17, R18, 0x1f;                                         /* 0x5800c0007d245c23 */
        /*0038*/                   ISETP.GE.U32.X.AND P0, PT, R17, c[0x0][0x144], PT;          /* 0x1b0e40051111dc43 */
                                                                                               /* 0x220042b0420042e7 */
        /*0048*/               @P0 EXIT;                                                       /* 0x80000000000001e7 */
        /*0050*/                   IADD R18.CC, R18, c[0x0][0x150];                            /* 0x4801400541249c03 */
        /*0058*/                   MOV32I R0, 0x3;                                             /* 0x180000000c001de2 */
        /*0060*/                   MOV32I R44, 0x0;                                            /* 0x18000000000b1de2 */
        /*0068*/                   MOV32I R39, 0x1f40;                                         /* 0x1800007d0009dde2 */
        /*0070*/                   IADD.X R17, R17, c[0x0][0x154];                             /* 0x4800400551145c43 */
        /*0078*/                   IADD32I RZ.CC, R0, -0x20;                                   /* 0x0fffffff800fdc02 */
                                                                                               /* 0x2202404230422047 */
        /*0088*/                   MOV32I R43, 0x0;                                            /* 0x18000000000adde2 */
        /*0090*/                   MOV R40, RZ;                                                /* 0x28000000fc0a1de4 */
        /*0098*/                   SHL R38, R18, -0x1d;                                        /* 0x6000ffff8d299c03 */
        /*00a0*/                   MOV32I R16, 0x4;                                            /* 0x1800000010041de2 */
        /*00a8*/                   SHR.U32 R3, R18, 0x1d;                                      /* 0x5800c0007520dc03 */
        /*00b0*/                   MOV R34, RZ;                                                /* 0x28000000fc089de4 */
        /*00b8*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
                                                                                               /* 0x2272004200422047 */
        /*00c8*/                   IADD32I R4.CC, R44, 0x10;                                   /* 0x0c00000042c11c02 */
        /*00d0*/                   MOV32I R15, 0x3;                                            /* 0x180000000c03dde2 */
        /*00d8*/                   SHL R0, R17, 0x3;                                           /* 0x6000c0000d101c03 */
        /*00e0*/                   MOV R35, RZ;                                                /* 0x28000000fc08dde4 */
        /*00e8*/                   MOV32I R14, 0x2;                                            /* 0x1800000008039de2 */
        /*00f0*/                   MOV R36, RZ;                                                /* 0x28000000fc091de4 */
        /*00f8*/                   MOV R37, RZ;                                                /* 0x28000000fc095de4 */
                                                                                               /* 0x2202c042b0420047 */
        /*0108*/                   SHL R41, R18, 0x3;                                          /* 0x6000c0000d2a5c03 */
        /*0110*/                   IADD.X R5, RZ, R43;                                         /* 0x48000000aff15c43 */
        /*0118*/               @P0 IADD R38, R0, R3;                                           /* 0x480000000c098003 */
        /*0120*/                   MOV32I R0, 0x1;                                             /* 0x1800000004001de2 */
        /*0128*/                   IADD RZ.CC, -R0, R18;                                       /* 0x48010000480fde03 */
        /*0130*/                   SSY 0x5c0;                                                  /* 0x6000001220001c07 */
        /*0138*/                   ISETP.EQ.X.AND P1, PT, R17, R37, PT;                        /* 0x190e00009513dc63 */
                                                                                               /* 0x220042b042d042c7 */
        /*0148*/                   IADD32I R39.CC, R39, -0x4;                                  /* 0x0ffffffff279dc02 */
        /*0150*/                   IADD32I.X R40, R40, -0x1;                                   /* 0x0bfffffffe8a1c42 */
        /*0158*/                   IADD RZ.CC, -RZ, R39;                                       /* 0x480100009fffde03 */
        /*0160*/                   ISETP.NE.X.AND P0, PT, R40, RZ, PT;                         /* 0x1a8e0000fe81dc63 */
        /*0168*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0170*/                   IADD32I R7.CC, R44, 0x10;                                   /* 0x0c00000042c1dc02 */
        /*0178*/                   SHL R3, R37, 0x3;                                           /* 0x6000c0000e50dc03 */
                                                                                               /* 0x2202c0420042b047 */
        /*0188*/                   MOV32I R2, 0x3;                                             /* 0x180000000c009de2 */
        /*0190*/                   MOV32I R22, 0x0;                                            /* 0x1800000000059de2 */
        /*0198*/                   IADD.X R9, RZ, R43;                                         /* 0x48000000aff25c43 */
        /*01a0*/                   SHR.U32 R6, R0, 0x1d;                                       /* 0x5800c00074019c03 */
        /*01a8*/                   IADD R10.CC, R7, R41;                                       /* 0x48010000a4729c03 */
        /*01b0*/                   MOV32I R23, 0x3fe00000;                                     /* 0x18ff80000005dde2 */
        /*01b8*/                   IADD.X R11, R9, R38;                                        /* 0x480000009892dc43 */
                                                                                               /* 0x227202b202b20047 */
        /*01c8*/                   IADD32I RZ.CC, R2, -0x20;                                   /* 0x0fffffff802fdc02 */
        /*01d0*/                   SSY 0x328;                                                  /* 0x6000000540001c07 */
        /*01d8*/                   SHL R2, R0, -0x1d;                                          /* 0x6000ffff8c009c03 */
        /*01e0*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
        /*01e8*/                   IMAD.U32.U32 R8.CC, R0, 0x8, R7;                            /* 0x200fc00020021c03 */
        /*01f0*/               @P1 IADD R2, R3, R6;                                            /* 0x4800000018308403 */
        /*01f8*/                   MOV R3, R11;                                                /* 0x280000002c00dde4 */
                                                                                               /* 0x2273f23232804287 */
        /*0208*/                   IADD.X R9, R2, R9;                                          /* 0x4800000024225c43 */
        /*0210*/                   MOV R2, R10;                                                /* 0x2800000028009de4 */
        /*0218*/                   LD.E.64 R12, [R8+0x1f400];                                  /* 0x840007d000831ca5 */
        /*0220*/                   LD.E.64 R26, [R2+0x1f400];                                  /* 0x840007d000269ca5 */
        /*0228*/                   LD.E.64 R28, [R2+0xfa00];                                   /* 0x840003e800271ca5 */
        /*0230*/                   LD.E.64 R6, [R8+0xfa00];                                    /* 0x840003e800819ca5 */
        /*0238*/                   DADD R26, -R26, R12;                                        /* 0x4800000031a69e01 */
                                                                                               /* 0x2292828042929237 */
        /*0248*/                   DADD R28, -R28, R6;                                         /* 0x4800000019c71e01 */
        /*0250*/                   DMUL R6, R26, R26;                                          /* 0x5000000069a19c01 */
        /*0258*/                   DFMA R6, R28, R28, R6;                                      /* 0x200c000071c19c01 */
        /*0260*/                   IADD32I R24, R7, -0x3500000;                                /* 0x0bf2c00000761c02 */
        /*0268*/                   MUFU.RSQ64H R11, R7;                                        /* 0xc80000001c72dc00 */
        /*0270*/                   MOV R10, R24;                                               /* 0x2800000060029de4 */
        /*0278*/                   DMUL R12, R10, R10;                                         /* 0x5000000028a31c01 */
                                                                                               /* 0x2202028202923297 */
        /*0288*/                   DFMA R12, -R6, R12, c[0x2][0x0];                            /* 0x2018880000631e01 */
        /*0290*/                   DFMA R8, R12, 0.375, R22;                                   /* 0x202ccff600c21c01 */
        /*0298*/                   DMUL R12, R10, R12;                                         /* 0x5000000030a31c01 */
        /*02a0*/                   DFMA R20, R8, R12, R10;                                     /* 0x2014000030851c01 */
        /*02a8*/                   ISETP.LT.U32.AND P1, PT, R24, c[0x2][0x8], PT;              /* 0x188e48002183dc03 */
        /*02b0*/                   DMUL R10, R6, R20;                                          /* 0x5000000050629c01 */
        /*02b8*/                   IADD32I R19, R21, -0x100000;                                /* 0x0bffc0000154dc02 */
                                                                                               /* 0x220042b202820277 */
        /*02c8*/                   MOV R8, R20;                                                /* 0x2800000050021de4 */
        /*02d0*/                   DFMA R12, -R10, R10, R6;                                    /* 0x200c000028a31e01 */
        /*02d8*/                   MOV R9, R19;                                                /* 0x280000004c025de4 */
        /*02e0*/                   DFMA R30, R12, R8, R10;                                     /* 0x2014000020c79c01 */
        /*02e8*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*02f0*/                   MOV R8, R20;                                                /* 0x2800000050021de4 */
        /*02f8*/                   MOV R20, R10;                                               /* 0x2800000028051de4 */
                                                                                               /* 0x22023042b2b2b047 */
        /*0308*/                   MOV R21, R11;                                               /* 0x280000002c055de4 */
        /*0310*/                   MOV R9, R24;                                                /* 0x2800000060025de4 */
        /*0318*/                   CAL 0x16f0;                                                 /* 0x5000004f40011c07 */
        /*0320*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0328*/                   LD.E.64 R6, [R2];                                           /* 0x8400000000219ca5 */
        /*0330*/                   DFMA R8, R30, R30, c[0x2][0x18];                            /* 0x203c880061e21c01 */
        /*0338*/                   LD.E.64 R10, [R4+0x8];                                      /* 0x8400000020429ca5 */
                                                                                               /* 0x2293f29292824047 */
        /*0348*/                   MOV32I R12, 0x1;                                            /* 0x1800000004031de2 */
        /*0350*/                   SSY 0x400;                                                  /* 0x60000002a0001c07 */
        /*0358*/                   MUFU.RCP64H R13, R9;                                        /* 0xc800000018935c00 */
        /*0360*/                   DFMA R20, -R8, R12, c[0x2][0x0];                            /* 0x2018880000851e01 */
        /*0368*/                   DFMA R22, R20, R20, R20;                                    /* 0x2028000051459c01 */
        /*0370*/                   DFMA R12, R12, R22, R12;                                    /* 0x2018000058c31c01 */
        /*0378*/                   DMUL R6, R6, c[0x2][0x10];                                  /* 0x5000480040619c01 */
                                                                                               /* 0x22c2828202929297 */
        /*0388*/                   DMUL R10, R6, R10;                                          /* 0x5000000028629c01 */
        /*0390*/                   DMUL R6, R10, R12;                                          /* 0x5000000030a19c01 */
        /*0398*/                   DFMA R32, -R8, R6, R10;                                     /* 0x2014000018881e01 */
        /*03a0*/                   DFMA R32, R12, R32, R6;                                     /* 0x200c000080c81c01 */
        /*03a8*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0xc], PT;                 /* 0x160f480030bfdc80 */
        /*03b0*/                   FFMA R6, RZ, R9, R33;                                       /* 0x3042000027f19c00 */
        /*03b8*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
                                                                                               /* 0x22b2b042b2b042b7 */
        /*03c8*/               @P1 BRA CC.NEU, 0x3f8;                                          /* 0x40000000a00005a7 */
        /*03d0*/                   MOV R7, R10;                                                /* 0x280000002801dde4 */
        /*03d8*/                   MOV R6, R11;                                                /* 0x280000002c019de4 */
        /*03e0*/                   CAL 0x1398;                                                 /* 0x5000003ec0011c07 */
        /*03e8*/                   MOV R32, R10;                                               /* 0x2800000028081de4 */
        /*03f0*/                   MOV R33, R11;                                               /* 0x280000002c085de4 */
        /*03f8*/                   MUFU.RCP64H.S R9, R31;                                      /* 0xc800000019f25c10 */
                                                                                               /* 0x2292929292928047 */
        /*0408*/                   MOV32I R8, 0x1;                                             /* 0x1800000004021de2 */
        /*0410*/                   DMUL R28, R28, R32;                                         /* 0x5000000081c71c01 */
        /*0418*/                   DFMA R6, -R30, R8, c[0x2][0x0];                             /* 0x2010880001e19e01 */
        /*0420*/                   DFMA R6, R6, R6, R6;                                        /* 0x200c000018619c01 */
        /*0428*/                   DFMA R8, R8, R6, R8;                                        /* 0x2010000018821c01 */
        /*0430*/                   DMUL R10, R28, R8;                                          /* 0x5000000021c29c01 */
        /*0438*/                   DFMA R6, -R30, R10, R28;                                    /* 0x2038000029e19e01 */
                                                                                               /* 0x22b2028042804207 */
        /*0448*/                   DFMA R8, R8, R6, R10;                                       /* 0x2014000018821c01 */
        /*0450*/                   FSET.GTU.AND RZ.CC, |R29|, c[0x2][0xc], PT;                 /* 0x160f480031dfdc80 */
        /*0458*/                   MOV R10, R30;                                               /* 0x2800000078029de4 */
        /*0460*/                   FFMA R6, RZ, R31, R9;                                       /* 0x301200007ff19c00 */
        /*0468*/                   SSY 0x4d8;                                                  /* 0x60000001a0001c07 */
        /*0470*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
        /*0478*/                   MOV R11, R31;                                               /* 0x280000007c02dde4 */
                                                                                               /* 0x2202b2b0420042b7 */
        /*0488*/               @P1 BRA CC.NEU, 0x4d0;                                          /* 0x40000001000005a7 */
        /*0490*/                   MOV R7, R28;                                                /* 0x280000007001dde4 */
        /*0498*/                   MOV R6, R29;                                                /* 0x2800000074019de4 */
        /*04a0*/                   MOV R8, R10;                                                /* 0x2800000028021de4 */
        /*04a8*/                   MOV R9, R11;                                                /* 0x280000002c025de4 */
        /*04b0*/                   CAL 0x1398;                                                 /* 0x5000003b80011c07 */
        /*04b8*/                   MOV R8, R10;                                                /* 0x2800000028021de4 */
                                                                                               /* 0x229280420042b2b7 */
        /*04c8*/                   MOV R9, R11;                                                /* 0x280000002c025de4 */
        /*04d0*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*04d8*/                   LD.E.64 R6, [R2+0x4e200];                                   /* 0x8400138800219ca5 */
        /*04e0*/                   MUFU.RCP64H R11, R31;                                       /* 0xc800000019f2dc00 */
        /*04e8*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*04f0*/                   DMUL R26, R26, R32;                                         /* 0x5000000081a69c01 */
        /*04f8*/                   DFMA R12, -R30, R10, c[0x2][0x0];                           /* 0x2014880001e31e01 */
                                                                                               /* 0x23f0420292929297 */
        /*0508*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*0510*/                   DFMA R22, R10, R12, R10;                                    /* 0x2014000030a59c01 */
        /*0518*/                   DMUL R12, R22, R26;                                         /* 0x5000000069631c01 */
        /*0520*/                   DFMA R10, -R30, R12, R26;                                   /* 0x2034000031e29e01 */
        /*0528*/                   DFMA R10, R22, R10, R12;                                    /* 0x2018000029629c01 */
        /*0530*/                   FSET.GTU.AND RZ.CC, |R27|, c[0x2][0xc], PT;                 /* 0x160f480031bfdc80 */
        /*0538*/                   SSY 0x5a8;                                                  /* 0x60000001a0001c07 */
                                                                                               /* 0x22b042c042804207 */
        /*0548*/                   DADD R6, R8, R6;                                            /* 0x4800000018819c01 */
        /*0550*/                   FFMA R8, RZ, R31, R11;                                      /* 0x301600007ff21c00 */
        /*0558*/                   MOV R9, R31;                                                /* 0x280000007c025de4 */
        /*0560*/                   ST.E.64 [R2+0x4e200], R6;                                   /* 0x9400138800219ca5 */
        /*0568*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*0570*/                   MOV R8, R30;                                                /* 0x2800000078021de4 */
        /*0578*/               @P1 BRA CC.NEU, 0x5a0;                                          /* 0x40000000800005a7 */
                                                                                               /* 0x22b293f2b2b2b047 */
        /*0588*/                   MOV R7, R26;                                                /* 0x280000006801dde4 */
        /*0590*/                   MOV R6, R27;                                                /* 0x280000006c019de4 */
        /*0598*/                   CAL 0x1398;                                                 /* 0x50000037e0011c07 */
        /*05a0*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*05a8*/                   LD.E.64 R6, [R2+0x5dc00];                                   /* 0x8400177000219ca5 */
        /*05b0*/                   DADD R6, R10, R6;                                           /* 0x4800000018a19c01 */
        /*05b8*/                   ST.E.64.S [R2+0x5dc00], R6;                                 /* 0x9400177000219cb5 */
                                                                                               /* 0x2202c042b2c2c047 */
        /*05c8*/                   IADD RZ.CC, -R14, R18;                                      /* 0x4801000048efde03 */
        /*05d0*/                   SSY 0xa38;                                                  /* 0x6000001180001c07 */
        /*05d8*/                   ISETP.EQ.X.AND P1, PT, R17, R36, PT;                        /* 0x190e00009113dc63 */
        /*05e0*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*05e8*/                   IADD32I R8.CC, R44, 0x10;                                   /* 0x0c00000042c21c02 */
        /*05f0*/                   MOV32I R2, 0x3;                                             /* 0x180000000c009de2 */
        /*05f8*/                   IADD.X R9, RZ, R43;                                         /* 0x48000000aff25c43 */
                                                                                               /* 0x224230423042c047 */
        /*0608*/                   IADD R7.CC, R8, R41;                                        /* 0x48010000a481dc03 */
        /*0610*/                   SHL R3, R36, 0x3;                                           /* 0x6000c0000e40dc03 */
        /*0618*/                   IADD.X R10, R9, R38;                                        /* 0x4800000098929c43 */
        /*0620*/                   IADD32I RZ.CC, R2, -0x20;                                   /* 0x0fffffff802fdc02 */
        /*0628*/                   SHL R2, R14, -0x1d;                                         /* 0x6000ffff8ce09c03 */
        /*0630*/                   MOV32I R22, 0x0;                                            /* 0x1800000000059de2 */
        /*0638*/                   SHR.U32 R6, R14, 0x1d;                                      /* 0x5800c00074e19c03 */
                                                                                               /* 0x22828042b0420047 */
        /*0648*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
        /*0650*/                   MOV32I R23, 0x3fe00000;                                     /* 0x18ff80000005dde2 */
        /*0658*/                   IMAD.U32.U32 R8.CC, R14, 0x8, R8;                           /* 0x2011c00020e21c03 */
        /*0660*/                   SSY 0x7a0;                                                  /* 0x60000004e0001c07 */
        /*0668*/               @P1 IADD R2, R3, R6;                                            /* 0x4800000018308403 */
        /*0670*/                   MOV R3, R10;                                                /* 0x280000002800dde4 */
        /*0678*/                   IADD.X R9, R2, R9;                                          /* 0x4800000024225c43 */
                                                                                               /* 0x223273f232328047 */
        /*0688*/                   MOV R2, R7;                                                 /* 0x280000001c009de4 */
        /*0690*/                   LD.E.64 R12, [R8+0x1f400];                                  /* 0x840007d000831ca5 */
        /*0698*/                   LD.E.64 R26, [R2+0x1f400];                                  /* 0x840007d000269ca5 */
        /*06a0*/                   LD.E.64 R28, [R2+0xfa00];                                   /* 0x840003e800271ca5 */
        /*06a8*/                   LD.E.64 R6, [R8+0xfa00];                                    /* 0x840003e800819ca5 */
        /*06b0*/                   DADD R26, -R26, R12;                                        /* 0x4800000031a69e01 */
        /*06b8*/                   DADD R28, -R28, R6;                                         /* 0x4800000019c71e01 */
                                                                                               /* 0x2292928280429297 */
        /*06c8*/                   DMUL R6, R26, R26;                                          /* 0x5000000069a19c01 */
        /*06d0*/                   DFMA R6, R28, R28, R6;                                      /* 0x200c000071c19c01 */
        /*06d8*/                   IADD32I R24, R7, -0x3500000;                                /* 0x0bf2c00000761c02 */
        /*06e0*/                   MUFU.RSQ64H R11, R7;                                        /* 0xc80000001c72dc00 */
        /*06e8*/                   MOV R10, R24;                                               /* 0x2800000060029de4 */
        /*06f0*/                   DMUL R12, R10, R10;                                         /* 0x5000000028a31c01 */
        /*06f8*/                   DFMA R12, -R6, R12, c[0x2][0x0];                            /* 0x2018880000631e01 */
                                                                                               /* 0x2280420282029237 */
        /*0708*/                   DFMA R8, R12, 0.375, R22;                                   /* 0x202ccff600c21c01 */
        /*0710*/                   DMUL R12, R10, R12;                                         /* 0x5000000030a31c01 */
        /*0718*/                   DFMA R20, R8, R12, R10;                                     /* 0x2014000030851c01 */
        /*0720*/                   ISETP.LT.U32.AND P1, PT, R24, c[0x2][0x8], PT;              /* 0x188e48002183dc03 */
        /*0728*/                   DMUL R10, R6, R20;                                          /* 0x5000000050629c01 */
        /*0730*/                   IADD32I R19, R21, -0x100000;                                /* 0x0bffc0000154dc02 */
        /*0738*/                   MOV R8, R20;                                                /* 0x2800000050021de4 */
                                                                                               /* 0x22020042b2028207 */
        /*0748*/                   DFMA R12, -R10, R10, R6;                                    /* 0x200c000028a31e01 */
        /*0750*/                   MOV R9, R19;                                                /* 0x280000004c025de4 */
        /*0758*/                   DFMA R30, R12, R8, R10;                                     /* 0x2014000020c79c01 */
        /*0760*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0768*/                   MOV R8, R20;                                                /* 0x2800000050021de4 */
        /*0770*/                   MOV R20, R10;                                               /* 0x2800000028051de4 */
        /*0778*/                   MOV R21, R11;                                               /* 0x280000002c055de4 */
                                                                                               /* 0x2250423042b2b2b7 */
        /*0788*/                   MOV R9, R24;                                                /* 0x2800000060025de4 */
        /*0790*/                   CAL 0x16f0;                                                 /* 0x5000003d60011c07 */
        /*0798*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*07a0*/                   LD.E.64 R6, [R2];                                           /* 0x8400000000219ca5 */
        /*07a8*/                   DFMA R8, R30, R30, c[0x2][0x18];                            /* 0x203c880061e21c01 */
        /*07b0*/                   LD.E.64 R10, [R4+0x10];                                     /* 0x8400000040429ca5 */
        /*07b8*/                   MOV32I R12, 0x1;                                            /* 0x1800000004031de2 */
                                                                                               /* 0x229293f292928047 */
        /*07c8*/                   MUFU.RCP64H R13, R9;                                        /* 0xc800000018935c00 */
        /*07d0*/                   SSY 0x878;                                                  /* 0x6000000280001c07 */
        /*07d8*/                   DFMA R20, -R8, R12, c[0x2][0x0];                            /* 0x2018880000851e01 */
        /*07e0*/                   DFMA R22, R20, R20, R20;                                    /* 0x2028000051459c01 */
        /*07e8*/                   DFMA R12, R12, R22, R12;                                    /* 0x2018000058c31c01 */
        /*07f0*/                   DMUL R6, R6, c[0x2][0x10];                                  /* 0x5000480040619c01 */
        /*07f8*/                   DMUL R10, R6, R10;                                          /* 0x5000000028629c01 */
                                                                                               /* 0x22b2c28282029297 */
        /*0808*/                   DMUL R6, R10, R12;                                          /* 0x5000000030a19c01 */
        /*0810*/                   DFMA R32, -R8, R6, R10;                                     /* 0x2014000018881e01 */
        /*0818*/                   DFMA R32, R12, R32, R6;                                     /* 0x200c000080c81c01 */
        /*0820*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0xc], PT;                 /* 0x160f480030bfdc80 */
        /*0828*/                   FFMA R6, RZ, R9, R33;                                       /* 0x3042000027f19c00 */
        /*0830*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
        /*0838*/               @P1 BRA CC.NEU, 0x870;                                          /* 0x40000000c00005a7 */
                                                                                               /* 0x2282b2b042b2b047 */
        /*0848*/                   MOV R7, R10;                                                /* 0x280000002801dde4 */
        /*0850*/                   MOV R6, R11;                                                /* 0x280000002c019de4 */
        /*0858*/                   CAL 0x1398;                                                 /* 0x5000002ce0011c07 */
        /*0860*/                   MOV R32, R10;                                               /* 0x2800000028081de4 */
        /*0868*/                   MOV R33, R11;                                               /* 0x280000002c085de4 */
        /*0870*/                   MUFU.RCP64H.S R9, R31;                                      /* 0xc800000019f25c10 */
        /*0878*/                   MOV32I R8, 0x1;                                             /* 0x1800000004021de2 */
                                                                                               /* 0x2202929292523297 */
        /*0888*/                   DFMA R6, -R30, R8, c[0x2][0x0];                             /* 0x2010880001e19e01 */
        /*0890*/                   DFMA R6, R6, R6, R6;                                        /* 0x200c000018619c01 */
        /*0898*/                   DMUL R28, R28, R32;                                         /* 0x5000000081c71c01 */
        /*08a0*/                   DFMA R8, R8, R6, R8;                                        /* 0x2010000018821c01 */
        /*08a8*/                   DMUL R10, R28, R8;                                          /* 0x5000000021c29c01 */
        /*08b0*/                   DFMA R6, -R30, R10, R28;                                    /* 0x2038000029e19e01 */
        /*08b8*/                   DFMA R8, R8, R6, R10;                                       /* 0x2014000018821c01 */
                                                                                               /* 0x22b042c280428047 */
        /*08c8*/                   FSET.GTU.AND RZ.CC, |R29|, c[0x2][0xc], PT;                 /* 0x160f480031dfdc80 */
        /*08d0*/                   MOV R10, R30;                                               /* 0x2800000078029de4 */
        /*08d8*/                   FFMA R6, RZ, R31, R9;                                       /* 0x301200007ff19c00 */
        /*08e0*/                   SSY 0x950;                                                  /* 0x60000001a0001c07 */
        /*08e8*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
        /*08f0*/                   MOV R11, R31;                                               /* 0x280000007c02dde4 */
        /*08f8*/               @P1 BRA CC.NEU, 0x940;                                          /* 0x40000001000005a7 */
                                                                                               /* 0x22b042b2b0420047 */
        /*0908*/                   MOV R7, R28;                                                /* 0x280000007001dde4 */
        /*0910*/                   MOV R6, R29;                                                /* 0x2800000074019de4 */
        /*0918*/                   MOV R8, R10;                                                /* 0x2800000028021de4 */
        /*0920*/                   MOV R9, R11;                                                /* 0x280000002c025de4 */
        /*0928*/                   CAL 0x1398;                                                 /* 0x50000029a0011c07 */
        /*0930*/                   MOV R8, R10;                                                /* 0x2800000028021de4 */
        /*0938*/                   MOV R9, R11;                                                /* 0x280000002c025de4 */
                                                                                               /* 0x22929280420042b7 */
        /*0948*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0950*/                   LD.E.64 R6, [R2+0x4e200];                                   /* 0x8400138800219ca5 */
        /*0958*/                   MUFU.RCP64H R11, R31;                                       /* 0xc800000019f2dc00 */
        /*0960*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*0968*/                   DMUL R26, R26, R32;                                         /* 0x5000000081a69c01 */
        /*0970*/                   DFMA R12, -R30, R10, c[0x2][0x0];                           /* 0x2014880001e31e01 */
        /*0978*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
                                                                                               /* 0x2203f04202929297 */
        /*0988*/                   DFMA R22, R10, R12, R10;                                    /* 0x2014000030a59c01 */
        /*0990*/                   DMUL R12, R22, R26;                                         /* 0x5000000069631c01 */
        /*0998*/                   DFMA R10, -R30, R12, R26;                                   /* 0x2034000031e29e01 */
        /*09a0*/                   DFMA R10, R22, R10, R12;                                    /* 0x2018000029629c01 */
        /*09a8*/                   FSET.GTU.AND RZ.CC, |R27|, c[0x2][0xc], PT;                 /* 0x160f480031bfdc80 */
        /*09b0*/                   SSY 0xa20;                                                  /* 0x60000001a0001c07 */
        /*09b8*/                   DADD R6, R8, R6;                                            /* 0x4800000018819c01 */
                                                                                               /* 0x2202b042c0428047 */
        /*09c8*/                   FFMA R8, RZ, R31, R11;                                      /* 0x301600007ff21c00 */
        /*09d0*/                   MOV R9, R31;                                                /* 0x280000007c025de4 */
        /*09d8*/                   ST.E.64 [R2+0x4e200], R6;                                   /* 0x9400138800219ca5 */
        /*09e0*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*09e8*/                   MOV R8, R30;                                                /* 0x2800000078021de4 */
        /*09f0*/               @P1 BRA CC.NEU, 0xa18;                                          /* 0x40000000800005a7 */
        /*09f8*/                   MOV R7, R26;                                                /* 0x280000006801dde4 */
                                                                                               /* 0x22c2b293f2b2b2b7 */
        /*0a08*/                   MOV R6, R27;                                                /* 0x280000006c019de4 */
        /*0a10*/                   CAL 0x1398;                                                 /* 0x5000002600011c07 */
        /*0a18*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0a20*/                   LD.E.64 R6, [R2+0x5dc00];                                   /* 0x8400177000219ca5 */
        /*0a28*/                   DADD R6, R10, R6;                                           /* 0x4800000018a19c01 */
        /*0a30*/                   ST.E.64.S [R2+0x5dc00], R6;                                 /* 0x9400177000219cb5 */
        /*0a38*/                   IADD RZ.CC, -R15, R18;                                      /* 0x4801000048ffde03 */
                                                                                               /* 0x22b0420042b2c047 */
        /*0a48*/                   ISETP.EQ.X.AND P1, PT, R17, R35, PT;                        /* 0x190e00008d13dc63 */
        /*0a50*/                   SSY 0xeb0;                                                  /* 0x6000001160001c07 */
        /*0a58*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0a60*/                   IADD32I R8.CC, R44, 0x10;                                   /* 0x0c00000042c21c02 */
        /*0a68*/                   SHL R3, R35, 0x3;                                           /* 0x6000c0000e30dc03 */
        /*0a70*/                   MOV32I R2, 0x3;                                             /* 0x180000000c009de2 */
        /*0a78*/                   MOV32I R22, 0x0;                                            /* 0x1800000000059de2 */
                                                                                               /* 0x22823042c0420047 */
        /*0a88*/                   IADD.X R9, RZ, R43;                                         /* 0x48000000aff25c43 */
        /*0a90*/                   SHR.U32 R6, R15, 0x1d;                                      /* 0x5800c00074f19c03 */
        /*0a98*/                   IADD R7.CC, R8, R41;                                        /* 0x48010000a481dc03 */
        /*0aa0*/                   MOV32I R23, 0x3fe00000;                                     /* 0x18ff80000005dde2 */
        /*0aa8*/                   IADD.X R10, R9, R38;                                        /* 0x4800000098929c43 */
        /*0ab0*/                   IADD32I RZ.CC, R2, -0x20;                                   /* 0x0fffffff802fdc02 */
        /*0ab8*/                   SHL R2, R15, -0x1d;                                         /* 0x6000ffff8cf09c03 */
                                                                                               /* 0x2280427202b20047 */
        /*0ac8*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
        /*0ad0*/                   SSY 0xc18;                                                  /* 0x6000000500001c07 */
        /*0ad8*/                   IMAD.U32.U32 R8.CC, R15, 0x8, R8;                           /* 0x2011c00020f21c03 */
        /*0ae0*/               @P1 IADD R2, R3, R6;                                            /* 0x4800000018308403 */
        /*0ae8*/                   MOV R3, R10;                                                /* 0x280000002800dde4 */
        /*0af0*/                   IADD.X R9, R2, R9;                                          /* 0x4800000024225c43 */
        /*0af8*/                   MOV R2, R7;                                                 /* 0x280000001c009de4 */
                                                                                               /* 0x22923273f2323237 */
        /*0b08*/                   LD.E.64 R26, [R2+0x1f400];                                  /* 0x840007d000269ca5 */
        /*0b10*/                   LD.E.64 R12, [R8+0x1f400];                                  /* 0x840007d000831ca5 */
        /*0b18*/                   LD.E.64 R28, [R2+0xfa00];                                   /* 0x840003e800271ca5 */
        /*0b20*/                   LD.E.64 R6, [R8+0xfa00];                                    /* 0x840003e800819ca5 */
        /*0b28*/                   DADD R26, -R26, R12;                                        /* 0x4800000031a69e01 */
        /*0b30*/                   DADD R28, -R28, R6;                                         /* 0x4800000019c71e01 */
        /*0b38*/                   DMUL R6, R26, R26;                                          /* 0x5000000069a19c01 */
                                                                                               /* 0x2232929282804297 */
        /*0b48*/                   DFMA R6, R28, R28, R6;                                      /* 0x200c000071c19c01 */
        /*0b50*/                   IADD32I R24, R7, -0x3500000;                                /* 0x0bf2c00000761c02 */
        /*0b58*/                   MUFU.RSQ64H R11, R7;                                        /* 0xc80000001c72dc00 */
        /*0b60*/                   MOV R10, R24;                                               /* 0x2800000060029de4 */
        /*0b68*/                   DMUL R12, R10, R10;                                         /* 0x5000000028a31c01 */
        /*0b70*/                   DFMA R12, -R6, R12, c[0x2][0x0];                            /* 0x2018880000631e01 */
        /*0b78*/                   DFMA R8, R12, 0.375, R22;                                   /* 0x202ccff600c21c01 */
                                                                                               /* 0x2202804202820297 */
        /*0b88*/                   DMUL R12, R10, R12;                                         /* 0x5000000030a31c01 */
        /*0b90*/                   DFMA R20, R8, R12, R10;                                     /* 0x2014000030851c01 */
        /*0b98*/                   ISETP.LT.U32.AND P1, PT, R24, c[0x2][0x8], PT;              /* 0x188e48002183dc03 */
        /*0ba0*/                   DMUL R10, R6, R20;                                          /* 0x5000000050629c01 */
        /*0ba8*/                   IADD32I R19, R21, -0x100000;                                /* 0x0bffc0000154dc02 */
        /*0bb0*/                   MOV R8, R20;                                                /* 0x2800000050021de4 */
        /*0bb8*/                   DFMA R12, -R10, R10, R6;                                    /* 0x200c000028a31e01 */
                                                                                               /* 0x22b0420042b20287 */
        /*0bc8*/                   MOV R9, R19;                                                /* 0x280000004c025de4 */
        /*0bd0*/                   DFMA R30, R12, R8, R10;                                     /* 0x2014000020c79c01 */
        /*0bd8*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0be0*/                   MOV R8, R20;                                                /* 0x2800000050021de4 */
        /*0be8*/                   MOV R20, R10;                                               /* 0x2800000028051de4 */
        /*0bf0*/                   MOV R21, R11;                                               /* 0x280000002c055de4 */
        /*0bf8*/                   MOV R9, R24;                                                /* 0x2800000060025de4 */
                                                                                               /* 0x228250423042b2b7 */
        /*0c08*/                   CAL 0x16f0;                                                 /* 0x5000002b80011c07 */
        /*0c10*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0c18*/                   LD.E.64 R6, [R2];                                           /* 0x8400000000219ca5 */
        /*0c20*/                   DFMA R8, R30, R30, c[0x2][0x18];                            /* 0x203c880061e21c01 */
        /*0c28*/                   LD.E.64 R10, [R4+0x18];                                     /* 0x8400000060429ca5 */
        /*0c30*/                   MOV32I R12, 0x1;                                            /* 0x1800000004031de2 */
        /*0c38*/                   MUFU.RCP64H R13, R9;                                        /* 0xc800000018935c00 */
                                                                                               /* 0x22929293f2029297 */
        /*0c48*/                   DFMA R20, -R8, R12, c[0x2][0x0];                            /* 0x2018880000851e01 */
        /*0c50*/                   DFMA R22, R20, R20, R20;                                    /* 0x2028000051459c01 */
        /*0c58*/                   DFMA R12, R12, R22, R12;                                    /* 0x2018000058c31c01 */
        /*0c60*/                   SSY 0xcf0;                                                  /* 0x6000000220001c07 */
        /*0c68*/                   DMUL R6, R6, c[0x2][0x10];                                  /* 0x5000480040619c01 */
        /*0c70*/                   DMUL R10, R6, R10;                                          /* 0x5000000028629c01 */
        /*0c78*/                   DMUL R6, R10, R12;                                          /* 0x5000000030a19c01 */
                                                                                               /* 0x2202b2c282820297 */
        /*0c88*/                   DFMA R32, -R8, R6, R10;                                     /* 0x2014000018881e01 */
        /*0c90*/                   DFMA R32, R12, R32, R6;                                     /* 0x200c000080c81c01 */
        /*0c98*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0xc], PT;                 /* 0x160f480030bfdc80 */
        /*0ca0*/                   FFMA R6, RZ, R9, R33;                                       /* 0x3042000027f19c00 */
        /*0ca8*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
        /*0cb0*/               @P1 BRA CC.NEU, 0xce8;                                          /* 0x40000000c00005a7 */
        /*0cb8*/                   MOV R7, R10;                                                /* 0x280000002801dde4 */
                                                                                               /* 0x228042b2b042b2b7 */
        /*0cc8*/                   MOV R6, R11;                                                /* 0x280000002c019de4 */
        /*0cd0*/                   CAL 0x1398;                                                 /* 0x5000001b00011c07 */
        /*0cd8*/                   MOV R32, R10;                                               /* 0x2800000028081de4 */
        /*0ce0*/                   MOV R33, R11;                                               /* 0x280000002c085de4 */
        /*0ce8*/                   MUFU.RCP64H.S R9, R31;                                      /* 0xc800000019f25c10 */
        /*0cf0*/                   MOV32I R8, 0x1;                                             /* 0x1800000004021de2 */
        /*0cf8*/                   DMUL R28, R28, R32;                                         /* 0x5000000081c71c01 */
                                                                                               /* 0x2282029292929297 */
        /*0d08*/                   DFMA R6, -R30, R8, c[0x2][0x0];                             /* 0x2010880001e19e01 */
        /*0d10*/                   DFMA R6, R6, R6, R6;                                        /* 0x200c000018619c01 */
        /*0d18*/                   DFMA R8, R8, R6, R8;                                        /* 0x2010000018821c01 */
        /*0d20*/                   DMUL R10, R28, R8;                                          /* 0x5000000021c29c01 */
        /*0d28*/                   DFMA R6, -R30, R10, R28;                                    /* 0x2038000029e19e01 */
        /*0d30*/                   DFMA R8, R8, R6, R10;                                       /* 0x2014000018821c01 */
        /*0d38*/                   FSET.GTU.AND RZ.CC, |R29|, c[0x2][0xc], PT;                 /* 0x160f480031dfdc80 */
                                                                                               /* 0x2202b042b2028047 */
        /*0d48*/                   FFMA R6, RZ, R31, R9;                                       /* 0x301200007ff19c00 */
        /*0d50*/                   SSY 0xdc0;                                                  /* 0x60000001a0001c07 */
        /*0d58*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
        /*0d60*/                   MOV R10, R30;                                               /* 0x2800000078029de4 */
        /*0d68*/                   MOV R11, R31;                                               /* 0x280000007c02dde4 */
        /*0d70*/               @P1 BRA CC.NEU, 0xdb8;                                          /* 0x40000001000005a7 */
        /*0d78*/                   MOV R7, R28;                                                /* 0x280000007001dde4 */
                                                                                               /* 0x22b2b042b2b20047 */
        /*0d88*/                   MOV R6, R29;                                                /* 0x2800000074019de4 */
        /*0d90*/                   MOV R8, R10;                                                /* 0x2800000028021de4 */
        /*0d98*/                   MOV R9, R11;                                                /* 0x280000002c025de4 */
        /*0da0*/                   CAL 0x1398;                                                 /* 0x50000017c0011c07 */
        /*0da8*/                   MOV R8, R10;                                                /* 0x2800000028021de4 */
        /*0db0*/                   MOV R9, R11;                                                /* 0x280000002c025de4 */
        /*0db8*/                   NOP.S;                                                      /* 0x4000000000001df4 */
                                                                                               /* 0x2292929280420047 */
        /*0dc8*/                   LD.E.64 R6, [R2+0x4e200];                                   /* 0x8400138800219ca5 */
        /*0dd0*/                   MUFU.RCP64H R11, R31;                                       /* 0xc800000019f2dc00 */
        /*0dd8*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*0de0*/                   DMUL R26, R26, R32;                                         /* 0x5000000081a69c01 */
        /*0de8*/                   DFMA R12, -R30, R10, c[0x2][0x0];                           /* 0x2014880001e31e01 */
        /*0df0*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*0df8*/                   DFMA R22, R10, R12, R10;                                    /* 0x2014000030a59c01 */
                                                                                               /* 0x228203f042029297 */
        /*0e08*/                   DMUL R12, R22, R26;                                         /* 0x5000000069631c01 */
        /*0e10*/                   DFMA R10, -R30, R12, R26;                                   /* 0x2034000031e29e01 */
        /*0e18*/                   DFMA R10, R22, R10, R12;                                    /* 0x2018000029629c01 */
        /*0e20*/                   FSET.GTU.AND RZ.CC, |R27|, c[0x2][0xc], PT;                 /* 0x160f480031bfdc80 */
        /*0e28*/                   SSY 0xe98;                                                  /* 0x60000001a0001c07 */
        /*0e30*/                   DADD R6, R8, R6;                                            /* 0x4800000018819c01 */
        /*0e38*/                   FFMA R8, RZ, R31, R11;                                      /* 0x301600007ff21c00 */
                                                                                               /* 0x22b042b042b20047 */
        /*0e48*/                   ST.E.64 [R2+0x4e200], R6;                                   /* 0x9400138800219ca5 */
        /*0e50*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*0e58*/                   MOV R8, R30;                                                /* 0x2800000078021de4 */
        /*0e60*/                   MOV R9, R31;                                                /* 0x280000007c025de4 */
        /*0e68*/               @P1 BRA CC.NEU, 0xe90;                                          /* 0x40000000800005a7 */
        /*0e70*/                   MOV R7, R26;                                                /* 0x280000006801dde4 */
        /*0e78*/                   MOV R6, R27;                                                /* 0x280000006c019de4 */
                                                                                               /* 0x22c042b293f2b2b7 */
        /*0e88*/                   CAL 0x1398;                                                 /* 0x5000001420011c07 */
        /*0e90*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*0e98*/                   LD.E.64 R6, [R2+0x5dc00];                                   /* 0x8400177000219ca5 */
        /*0ea0*/                   DADD R6, R10, R6;                                           /* 0x4800000018a19c01 */
        /*0ea8*/                   ST.E.64.S [R2+0x5dc00], R6;                                 /* 0x9400177000219cb5 */
        /*0eb0*/                   IADD RZ.CC, -R16, R18;                                      /* 0x48010000490fde03 */
        /*0eb8*/                   SSY 0x1328;                                                 /* 0x60000011a0001c07 */
                                                                                               /* 0x2202b0420042b2c7 */
        /*0ec8*/                   ISETP.EQ.X.AND P1, PT, R17, R34, PT;                        /* 0x190e00008913dc63 */
        /*0ed0*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*0ed8*/                   IADD32I R7.CC, R44, 0x10;                                   /* 0x0c00000042c1dc02 */
        /*0ee0*/                   SHL R3, R34, 0x3;                                           /* 0x6000c0000e20dc03 */
        /*0ee8*/                   MOV32I R2, 0x3;                                             /* 0x180000000c009de2 */
        /*0ef0*/                   MOV32I R22, 0x0;                                            /* 0x1800000000059de2 */
        /*0ef8*/                   IADD.X R9, RZ, R43;                                         /* 0x48000000aff25c43 */
                                                                                               /* 0x220280423042c047 */
        /*0f08*/                   IADD R10.CC, R7, R41;                                       /* 0x48010000a4729c03 */
        /*0f10*/                   SHR.U32 R6, R16, 0x1d;                                      /* 0x5800c00075019c03 */
        /*0f18*/                   IADD.X R11, R9, R38;                                        /* 0x480000009892dc43 */
        /*0f20*/                   IADD32I RZ.CC, R2, -0x20;                                   /* 0x0fffffff802fdc02 */
        /*0f28*/                   SHL R2, R16, -0x1d;                                         /* 0x6000ffff8d009c03 */
        /*0f30*/                   MOV32I R23, 0x3fe00000;                                     /* 0x18ff80000005dde2 */
        /*0f38*/                   CSETP.SFT.AND P1, PT, CC, PT;                               /* 0x040e00000153dc04 */
                                                                                               /* 0x228042827202b047 */
        /*0f48*/                   IMAD.U32.U32 R8.CC, R16, 0x8, R7;                           /* 0x200fc00021021c03 */
        /*0f50*/                   SSY 0x1090;                                                 /* 0x60000004e0001c07 */
        /*0f58*/               @P1 IADD R2, R3, R6;                                            /* 0x4800000018308403 */
        /*0f60*/                   MOV R3, R11;                                                /* 0x280000002c00dde4 */
        /*0f68*/                   IADD.X R9, R2, R9;                                          /* 0x4800000024225c43 */
        /*0f70*/                   MOV R2, R10;                                                /* 0x2800000028009de4 */
        /*0f78*/                   LD.E.64 R12, [R8+0x1f400];                                  /* 0x840007d000831ca5 */
                                                                                               /* 0x2292923273f23237 */
        /*0f88*/                   LD.E.64 R26, [R2+0x1f400];                                  /* 0x840007d000269ca5 */
        /*0f90*/                   LD.E.64 R28, [R2+0xfa00];                                   /* 0x840003e800271ca5 */
        /*0f98*/                   LD.E.64 R6, [R8+0xfa00];                                    /* 0x840003e800819ca5 */
        /*0fa0*/                   DADD R26, -R26, R12;                                        /* 0x4800000031a69e01 */
        /*0fa8*/                   DADD R28, -R28, R6;                                         /* 0x4800000019c71e01 */
        /*0fb0*/                   DMUL R6, R26, R26;                                          /* 0x5000000069a19c01 */
        /*0fb8*/                   DFMA R6, R28, R28, R6;                                      /* 0x200c000071c19c01 */
                                                                                               /* 0x2292329292828047 */
        /*0fc8*/                   IADD32I R24, R7, -0x3500000;                                /* 0x0bf2c00000761c02 */
        /*0fd0*/                   MUFU.RSQ64H R11, R7;                                        /* 0xc80000001c72dc00 */
        /*0fd8*/                   MOV R10, R24;                                               /* 0x2800000060029de4 */
        /*0fe0*/                   DMUL R12, R10, R10;                                         /* 0x5000000028a31c01 */
        /*0fe8*/                   DFMA R12, -R6, R12, c[0x2][0x0];                            /* 0x2018880000631e01 */
        /*0ff0*/                   DFMA R8, R12, 0.375, R22;                                   /* 0x202ccff600c21c01 */
        /*0ff8*/                   DMUL R12, R10, R12;                                         /* 0x5000000030a31c01 */
                                                                                               /* 0x2282028042028207 */
        /*1008*/                   DFMA R20, R8, R12, R10;                                     /* 0x2014000030851c01 */
        /*1010*/                   ISETP.LT.U32.AND P1, PT, R24, c[0x2][0x8], PT;              /* 0x188e48002183dc03 */
        /*1018*/                   DMUL R10, R6, R20;                                          /* 0x5000000050629c01 */
        /*1020*/                   IADD32I R19, R21, -0x100000;                                /* 0x0bffc0000154dc02 */
        /*1028*/                   MOV R8, R20;                                                /* 0x2800000050021de4 */
        /*1030*/                   DFMA R12, -R10, R10, R6;                                    /* 0x200c000028a31e01 */
        /*1038*/                   MOV R9, R19;                                                /* 0x280000004c025de4 */
                                                                                               /* 0x22b2b0420042b207 */
        /*1048*/                   DFMA R30, R12, R8, R10;                                     /* 0x2014000020c79c01 */
        /*1050*/               @P1 NOP.S;                                                      /* 0x40000000000005f4 */
        /*1058*/                   MOV R8, R20;                                                /* 0x2800000050021de4 */
        /*1060*/                   MOV R20, R10;                                               /* 0x2800000028051de4 */
        /*1068*/                   MOV R21, R11;                                               /* 0x280000002c055de4 */
        /*1070*/                   MOV R9, R24;                                                /* 0x2800000060025de4 */
        /*1078*/                   CAL 0x16f0;                                                 /* 0x50000019c0011c07 */
                                                                                               /* 0x22804250423042b7 */
        /*1088*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*1090*/                   LD.E.64 R6, [R2];                                           /* 0x8400000000219ca5 */
        /*1098*/                   DFMA R8, R30, R30, c[0x2][0x18];                            /* 0x203c880061e21c01 */
        /*10a0*/                   LD.E.64 R10, [R4+0x20];                                     /* 0x8400000080429ca5 */
        /*10a8*/                   MOV32I R12, 0x1;                                            /* 0x1800000004031de2 */
        /*10b0*/                   MUFU.RCP64H R13, R9;                                        /* 0xc800000018935c00 */
        /*10b8*/                   SSY 0x1168;                                                 /* 0x60000002a0001c07 */
                                                                                               /* 0x2292929293f29297 */
        /*10c8*/                   DFMA R20, -R8, R12, c[0x2][0x0];                            /* 0x2018880000851e01 */
        /*10d0*/                   DFMA R22, R20, R20, R20;                                    /* 0x2028000051459c01 */
        /*10d8*/                   DFMA R12, R12, R22, R12;                                    /* 0x2018000058c31c01 */
        /*10e0*/                   DMUL R6, R6, c[0x2][0x10];                                  /* 0x5000480040619c01 */
        /*10e8*/                   DMUL R10, R6, R10;                                          /* 0x5000000028629c01 */
        /*10f0*/                   DMUL R6, R10, R12;                                          /* 0x5000000030a19c01 */
        /*10f8*/                   DFMA R32, -R8, R6, R10;                                     /* 0x2014000018881e01 */
                                                                                               /* 0x22b042b2c2828207 */
        /*1108*/                   DFMA R32, R12, R32, R6;                                     /* 0x200c000080c81c01 */
        /*1110*/                   FSET.GTU.AND RZ.CC, |R11|, c[0x2][0xc], PT;                 /* 0x160f480030bfdc80 */
        /*1118*/                   FFMA R6, RZ, R9, R33;                                       /* 0x3042000027f19c00 */
        /*1120*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
        /*1128*/               @P1 BRA CC.NEU, 0x1160;                                         /* 0x40000000c00005a7 */
        /*1130*/                   MOV R7, R10;                                                /* 0x280000002801dde4 */
        /*1138*/                   MOV R6, R11;                                                /* 0x280000002c019de4 */
                                                                                               /* 0x22928042b2b042b7 */
        /*1148*/                   CAL 0x1398;                                                 /* 0x5000000920011c07 */
        /*1150*/                   MOV R32, R10;                                               /* 0x2800000028081de4 */
        /*1158*/                   MOV R33, R11;                                               /* 0x280000002c085de4 */
        /*1160*/                   MUFU.RCP64H.S R9, R31;                                      /* 0xc800000019f25c10 */
        /*1168*/                   MOV32I R8, 0x1;                                             /* 0x1800000004021de2 */
        /*1170*/                   DMUL R28, R28, R32;                                         /* 0x5000000081c71c01 */
        /*1178*/                   DFMA R6, -R30, R8, c[0x2][0x0];                             /* 0x2010880001e19e01 */
                                                                                               /* 0x2280420292929297 */
        /*1188*/                   DFMA R6, R6, R6, R6;                                        /* 0x200c000018619c01 */
        /*1190*/                   DFMA R8, R8, R6, R8;                                        /* 0x2010000018821c01 */
        /*1198*/                   DMUL R10, R28, R8;                                          /* 0x5000000021c29c01 */
        /*11a0*/                   DFMA R6, -R30, R10, R28;                                    /* 0x2038000029e19e01 */
        /*11a8*/                   DFMA R8, R8, R6, R10;                                       /* 0x2014000018821c01 */
        /*11b0*/                   FSET.GTU.AND RZ.CC, |R29|, c[0x2][0xc], PT;                 /* 0x160f480031dfdc80 */
        /*11b8*/                   MOV R10, R30;                                               /* 0x2800000078029de4 */
                                                                                               /* 0x220042b042c28047 */
        /*11c8*/                   FFMA R6, RZ, R31, R9;                                       /* 0x301200007ff19c00 */
        /*11d0*/                   SSY 0x1238;                                                 /* 0x6000000180001c07 */
        /*11d8*/                   FSETP.GT.AND P1, PT, |R6|, c[0x2][0x20], PT;                /* 0x220e48008063dc80 */
        /*11e0*/                   MOV R11, R31;                                               /* 0x280000007c02dde4 */
        /*11e8*/               @P1 BRA CC.NEU, 0x1230;                                         /* 0x40000001000005a7 */
        /*11f0*/                   MOV R7, R28;                                                /* 0x280000007001dde4 */
        /*11f8*/                   MOV R6, R29;                                                /* 0x2800000074019de4 */
                                                                                               /* 0x2202b2b042b2b047 */
        /*1208*/                   MOV R8, R10;                                                /* 0x2800000028021de4 */
        /*1210*/                   MOV R9, R11;                                                /* 0x280000002c025de4 */
        /*1218*/                   CAL 0x1398;                                                 /* 0x50000005e0011c07 */
        /*1220*/                   MOV R8, R10;                                                /* 0x2800000028021de4 */
        /*1228*/                   MOV R9, R11;                                                /* 0x280000002c025de4 */
        /*1230*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*1238*/                   LD.E.64 R6, [R2+0x4e200];                                   /* 0x8400138800219ca5 */
                                                                                               /* 0x2292925232928047 */
        /*1248*/                   MUFU.RCP64H R11, R31;                                       /* 0xc800000019f2dc00 */
        /*1250*/                   MOV32I R10, 0x1;                                            /* 0x1800000004029de2 */
        /*1258*/                   DFMA R12, -R30, R10, c[0x2][0x0];                           /* 0x2014880001e31e01 */
        /*1260*/                   DFMA R12, R12, R12, R12;                                    /* 0x2018000030c31c01 */
        /*1268*/                   DMUL R26, R26, R32;                                         /* 0x5000000081a69c01 */
        /*1270*/                   DFMA R22, R10, R12, R10;                                    /* 0x2014000030a59c01 */
        /*1278*/                   DMUL R12, R22, R26;                                         /* 0x5000000069631c01 */
                                                                                               /* 0x22804203f0420297 */
        /*1288*/                   DFMA R10, -R30, R12, R26;                                   /* 0x2034000031e29e01 */
        /*1290*/                   DFMA R10, R22, R10, R12;                                    /* 0x2018000029629c01 */
        /*1298*/                   FSET.GTU.AND RZ.CC, |R27|, c[0x2][0xc], PT;                 /* 0x160f480031bfdc80 */
        /*12a0*/                   SSY 0x1310;                                                 /* 0x60000001a0001c07 */
        /*12a8*/                   DADD R6, R8, R6;                                            /* 0x4800000018819c01 */
        /*12b0*/                   FFMA R8, RZ, R31, R11;                                      /* 0x301600007ff21c00 */
        /*12b8*/                   MOV R9, R31;                                                /* 0x280000007c025de4 */
                                                                                               /* 0x22b2b042b042c047 */
        /*12c8*/                   ST.E.64 [R2+0x4e200], R6;                                   /* 0x9400138800219ca5 */
        /*12d0*/                   FSETP.GT.AND P1, PT, |R8|, c[0x2][0x20], PT;                /* 0x220e48008083dc80 */
        /*12d8*/                   MOV R8, R30;                                                /* 0x2800000078021de4 */
        /*12e0*/               @P1 BRA CC.NEU, 0x1300;                                         /* 0x40000000600005a7 */
        /*12e8*/                   MOV R7, R26;                                                /* 0x280000006801dde4 */
        /*12f0*/                   MOV R6, R27;                                                /* 0x280000006c019de4 */
        /*12f8*/                   CAL 0x1398;                                                 /* 0x5000000260011c07 */
                                                                                               /* 0x22c042c2b293f2b7 */
        /*1308*/                   NOP.S;                                                      /* 0x4000000000001df4 */
        /*1310*/                   LD.E.64 R6, [R2+0x5dc00];                                   /* 0x8400177000219ca5 */
        /*1318*/                   DADD R6, R10, R6;                                           /* 0x4800000018a19c01 */
        /*1320*/                   ST.E.64.S [R2+0x5dc00], R6;                                 /* 0x9400177000219cb5 */
        /*1328*/                   IADD32I R4.CC, R4, 0x20;                                    /* 0x0c00000080411c02 */
        /*1330*/                   IADD.X R5, RZ, R5;                                          /* 0x4800000017f15c43 */
        /*1338*/                   IADD32I R16.CC, R16, 0x4;                                   /* 0x0c00000011041c02 */
                                                                                               /* 0x2202c042c202c047 */
        /*1348*/                   IADD.X R34, RZ, R34;                                        /* 0x480000008bf89c43 */
        /*1350*/                   IADD32I R15.CC, R15, 0x4;                                   /* 0x0c00000010f3dc02 */
        /*1358*/                   IADD.X R35, RZ, R35;                                        /* 0x480000008ff8dc43 */
        /*1360*/                   IADD32I R14.CC, R14, 0x4;                                   /* 0x0c00000010e39c02 */
        /*1368*/                   IADD.X R36, RZ, R36;                                        /* 0x4800000093f91c43 */
        /*1370*/                   IADD32I R0.CC, R0, 0x4;                                     /* 0x0c00000010001c02 */
        /*1378*/                   IADD.X R37, RZ, R37;                                        /* 0x4800000097f95c43 */
                                                                                               /* 0x220270420042e2e7 */
        /*1388*/               @P0 BRA 0x128;                                                  /* 0x4003ffb6600001e7 */
        /*1390*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*1398*/                   LOP32I.AND R10, R9, 0x40000000;                             /* 0x3900000000929c02 */
        /*13a0*/                   LOP32I.AND R19, R6, 0x7f800000;                             /* 0x39fe00000064dc02 */
        /*13a8*/                   MOV32I R42, 0x1ff00000;                                     /* 0x187fc000000a9de2 */
        /*13b0*/                   PBK 0x16d8;                                                 /* 0x6800000c80001c07 */
        /*13b8*/                   ISETP.LT.U32.AND P1, PT, R10, c[0x2][0x24], PT;             /* 0x188e480090a3dc03 */
                                                                                               /* 0x22c042028042b047 */
        /*13c8*/                   MOV R10, RZ;                                                /* 0x28000000fc029de4 */
        /*13d0*/                   LOP.XOR R7, R7, R6;                                         /* 0x680000001871dc83 */
        /*13d8*/                   SEL R11, R42, c[0x2][0x28], !P1;                            /* 0x20124800a2a2dc04 */
        /*13e0*/                   LOP.XOR R6, R7, R6;                                         /* 0x6800000018719c83 */
        /*13e8*/                   DMUL R12, R8, R10;                                          /* 0x5000000028831c01 */
        /*13f0*/                   ISETP.LT.U32.AND P1, PT, R19, c[0x2][0x2c], PT;             /* 0x188e4800b133dc03 */
        /*13f8*/                   LOP.XOR R7, R7, R6;                                         /* 0x680000001871dc83 */
                                                                                               /* 0x2280420280428047 */
        /*1408*/                   MUFU.RCP64H R22, R13;                                       /* 0xc800000018d59c00 */
        /*1410*/                   SEL R20, R42, c[0x2][0x28], !P1;                            /* 0x20124800a2a51c04 */
        /*1418*/                   MOV R23, R22;                                               /* 0x280000005805dde4 */
        /*1420*/                   MOV32I R22, 0x1;                                            /* 0x1800000004059de2 */
        /*1428*/                   DFMA R24, -R12, R22, c[0x2][0x0];                           /* 0x202c880000c61e01 */
        /*1430*/                   MOV R21, R20;                                               /* 0x2800000050055de4 */
        /*1438*/                   MOV R20, RZ;                                                /* 0x28000000fc051de4 */
                                                                                               /* 0x22d2929292925237 */
        /*1448*/                   DFMA R24, R24, R24, R24;                                    /* 0x2030000061861c01 */
        /*1450*/                   DMUL R20, R6, R20;                                          /* 0x5000000050651c01 */
        /*1458*/                   DFMA R22, R22, R24, R22;                                    /* 0x202c000061659c01 */
        /*1460*/                   DMUL R24, R20, R22;                                         /* 0x5000000059461c01 */
        /*1468*/                   DFMA R12, -R12, R24, R20;                                   /* 0x2028000060c31e01 */
        /*1470*/                   DFMA R12, R22, R12, R24;                                    /* 0x2030000031631c01 */
        /*1478*/                   DSETP.LEU.AND P1, PT, |R12|, RZ, PT;                        /* 0x1d8e0000fcc3dc81 */
                                                                                               /* 0x22523282823042b7 */
        /*1488*/               @P1 BRA 0x1658;                                                 /* 0x40000007200005e7 */
        /*1490*/                   ISETP.GT.U32.AND P1, PT, R19, c[0x2][0x30], PT;             /* 0x1a0e4800c133dc03 */
        /*1498*/                   MOV R20, RZ;                                                /* 0x28000000fc051de4 */
        /*14a0*/                   DMUL R22, R10, R12;                                         /* 0x5000000030a59c01 */
        /*14a8*/                   SEL R21, R42, c[0x2][0x28], !P1;                            /* 0x20124800a2a55c04 */
        /*14b0*/                   DMUL R12, R12, R20;                                         /* 0x5000000050c31c01 */
        /*14b8*/                   DMUL R20, R20, R22;                                         /* 0x5000000059451c01 */
                                                                                               /* 0x22c27202d2925237 */
        /*14c8*/                   DMUL R10, R10, R12;                                         /* 0x5000000030a29c01 */
        /*14d0*/                   DFMA R12, R8, R20, -R6;                                     /* 0x200c000050831d01 */
        /*14d8*/                   DFMA R22, R8, R10, -R6;                                     /* 0x200c000028859d01 */
        /*14e0*/                   DSETP.GT.AND P1, PT, |R12|, |R22|, PT;                      /* 0x1a0e000058c3dcc1 */
        /*14e8*/                   SEL R19, R11, R21, P1;                                      /* 0x2002000054b4dc04 */
        /*14f0*/                   SEL R12, R10, R20, P1;                                      /* 0x2002000050a31c04 */
        /*14f8*/                   FSETP.GTU.AND P1, PT, |R19|, 1.469367938527859385e-39, PT;  /* 0x260ec0040133dc80 */
                                                                                               /* 0x226200420042b047 */
        /*1508*/                   MOV R13, R19;                                               /* 0x280000004c035de4 */
        /*1510*/               @P1 BRK;                                                        /* 0xa8000000000005e7 */
        /*1518*/                   FSETP.LT.AND P1, PT, |R7|, 1.5046327690525280102e-36, PT;   /* 0x208ec1000073dc80 */
        /*1520*/                   LOP32I.AND R21, R12, 0xfffffffe;                            /* 0x3bfffffff8c55c02 */
        /*1528*/                   MOV32I R11, 0x3ff00000;                                     /* 0x18ffc0000002dde2 */
        /*1530*/                   MOV R13, R19;                                               /* 0x280000004c035de4 */
        /*1538*/                   MOV R10, RZ;                                                /* 0x28000000fc029de4 */
                                                                                               /* 0x2280420232823047 */
        /*1548*/                   LOP32I.OR R20, R12, 0x1;                                    /* 0x3800000004c51c42 */
        /*1550*/                   MOV R12, R21;                                               /* 0x2800000054031de4 */
        /*1558*/                   SEL R11, R11, c[0x2][0x34], !P1;                            /* 0x20124800d0b2dc04 */
        /*1560*/                   DMUL R8, R8, R10;                                           /* 0x5000000028821c01 */
        /*1568*/                   DMUL R6, R6, R10;                                           /* 0x5000000028619c01 */
        /*1570*/                   MOV R10, R20;                                               /* 0x2800000050029de4 */
        /*1578*/                   MOV R11, R19;                                               /* 0x280000004c02dde4 */
                                                                                               /* 0x2202c04282d29237 */
        /*1588*/                   DFMA R12, R12, R8, -R6;                                     /* 0x200c000020c31d01 */
        /*1590*/                   DFMA R10, R10, R8, -R6;                                     /* 0x200c000020a29d01 */
        /*1598*/                   DSETP.GT.AND P1, PT, |R12|, |R10|, PT;                      /* 0x1a0e000028c3dcc1 */
        /*15a0*/                   SEL R10, R20, R21, P1;                                      /* 0x2002000055429c04 */
        /*15a8*/                   IADD32I R21.CC, R10, 0x1;                                   /* 0x0c00000004a55c02 */
        /*15b0*/                   LOP32I.AND R12, R10, 0x1;                                   /* 0x3800000004a31c02 */
        /*15b8*/                   IADD.X R20, RZ, R19;                                        /* 0x480000004ff51c43 */
                                                                                               /* 0x220202720042c207 */
        /*15c8*/                   IADD32I R11.CC, R10, -0x1;                                  /* 0x0ffffffffca2dc02 */
        /*15d0*/                   ISETP.EQ.U32.AND P1, PT, R12, 0x1, PT;                      /* 0x190ec00004c3dc03 */
        /*15d8*/                   IADD32I.X R12, R19, -0x1;                                   /* 0x0bfffffffd331c42 */
        /*15e0*/                   SEL R21, R10, R21, !P1;                                     /* 0x2012000054a55c04 */
        /*15e8*/                   SEL R20, R19, R20, !P1;                                     /* 0x2012000051351c04 */
        /*15f0*/                   SEL R19, R12, R19, !P1;                                     /* 0x201200004cc4dc04 */
        /*15f8*/                   SEL R10, R11, R10, !P1;                                     /* 0x2012000028b29c04 */
                                                                                               /* 0x2202d29252225207 */
        /*1608*/                   MOV R12, R21;                                               /* 0x2800000054031de4 */
        /*1610*/                   MOV R13, R20;                                               /* 0x2800000050035de4 */
        /*1618*/                   MOV R11, R19;                                               /* 0x280000004c02dde4 */
        /*1620*/                   DFMA R12, R8, R12, -R6;                                     /* 0x200c000030831d01 */
        /*1628*/                   DFMA R6, R8, R10, -R6;                                      /* 0x200c000028819d01 */
        /*1630*/                   DSETP.GT.AND P1, PT, |R12|, |R6|, PT;                       /* 0x1a0e000018c3dcc1 */
        /*1638*/                   SEL R12, R10, R21, P1;                                      /* 0x2002000054a31c04 */
                                                                                               /* 0x22d28042b2d2e047 */
        /*1648*/                   SEL R13, R19, R20, P1;                                      /* 0x2002000051335c04 */
        /*1650*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1658*/                   DSETP.EQ.AND P1, PT, R12, RZ, PT;                           /* 0x190e0000fcc3dc01 */
        /*1660*/               @P1 BRA 0x16c0;                                                 /* 0x40000001600005e7 */
        /*1668*/                   MUFU.RCP64H R11, R9;                                        /* 0xc80000001892dc00 */
        /*1670*/                   MOV R10, RZ;                                                /* 0x28000000fc029de4 */
        /*1678*/                   DSETP.GT.AND P2, PT, |R10|, RZ, PT;                         /* 0x1a0e0000fca5dc81 */
                                                                                               /* 0x22e20282027202d7 */
        /*1688*/              @!P2 DSETP.NEU.AND P1, PT, |R8|, +INF , PT;                      /* 0x1e8edffc0083e881 */
        /*1690*/              @!P2 SEL R8, R8, R10, P1;                                        /* 0x2002000028822804 */
        /*1698*/              @!P2 SEL R9, R9, R11, P1;                                        /* 0x200200002c926804 */
        /*16a0*/              @!P2 MOV R10, R8;                                                /* 0x280000002002a9e4 */
        /*16a8*/              @!P2 MOV R11, R9;                                                /* 0x280000002402e9e4 */
        /*16b0*/                   DMUL R12, R6, R10;                                          /* 0x5000000028631c01 */
        /*16b8*/                   BRK;                                                        /* 0xa800000000001de7 */
                                                                                               /* 0x220042e20042b207 */
        /*16c8*/                   DMUL R12, R6, R8;                                           /* 0x5000000020631c01 */
        /*16d0*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*16d8*/                   MOV R10, R12;                                               /* 0x2800000030029de4 */
        /*16e0*/                   MOV R11, R13;                                               /* 0x280000003402dde4 */
        /*16e8*/                   RET;                                                        /* 0x9000000000001de7 */
        /*16f0*/                   ISETP.LT.U32.AND P1, PT, R9, c[0x2][0x38], PT;              /* 0x188e4800e093dc03 */
        /*16f8*/                   MOV R10, R12;                                               /* 0x2800000030029de4 */
                                                                                               /* 0x2282c292b2a20047 */
        /*1708*/                   MOV R11, R13;                                               /* 0x280000003402dde4 */
        /*1710*/                   MOV R9, R19;                                                /* 0x280000004c025de4 */
        /*1718*/                   PBK 0x1860;                                                 /* 0x6800000500001c07 */
        /*1720*/               @P1 BRA 0x1770;                                                 /* 0x40000001200005e7 */
        /*1728*/                   DFMA.RM R8, R10, R8, R20;                                   /* 0x20a8000020a21c01 */
        /*1730*/                   IADD32I R10.CC, R8, 0x1;                                    /* 0x0c00000004829c02 */
        /*1738*/                   IADD.X R11, RZ, R9;                                         /* 0x4800000027f2dc43 */
                                                                                               /* 0x22b2d2e04202d297 */
        /*1748*/                   DFMA.RP R6, -R8, R10, R6;                                   /* 0x210c000028819e01 */
        /*1750*/                   DSETP.GT.AND P1, PT, R6, RZ, PT;                            /* 0x1a0e0000fc63dc01 */
        /*1758*/                   SEL R8, R10, R8, P1;                                        /* 0x2002000020a21c04 */
        /*1760*/                   SEL R9, R11, R9, P1;                                        /* 0x2002000024b25c04 */
        /*1768*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1770*/                   DSETP.EQU.AND P1, PT, R6, RZ, PT;                           /* 0x1d0e0000fc63dc01 */
        /*1778*/               @P1 BRA 0x1850;                                                 /* 0x40000003400005e7 */
                                                                                               /* 0x2202b2c2b20042c7 */
        /*1788*/                   ISETP.LT.AND P1, PT, R7, RZ, PT;                            /* 0x188e0000fc73dc23 */
        /*1790*/               @P1 MOV32I R8, 0x0;                                             /* 0x18000000000205e2 */
        /*1798*/               @P1 MOV32I R9, 0xfff80000;                                      /* 0x1bffe000000245e2 */
        /*17a0*/               @P1 BRK;                                                        /* 0xa8000000000005e7 */
        /*17a8*/                   ISETP.GT.AND P1, PT, R7, c[0x2][0x3c], PT;                  /* 0x1a0e4800f073dc23 */
        /*17b0*/               @P1 BRA 0x1850;                                                 /* 0x40000002600005e7 */
        /*17b8*/                   DMUL R6, R6, 8.11296384146066816958e+31;                    /* 0x5000d1a400619c01 */
                                                                                               /* 0x2232929280428047 */
        /*17c8*/                   MOV R8, RZ;                                                 /* 0x28000000fc021de4 */
        /*17d0*/                   MOV32I R10, 0x0;                                            /* 0x1800000000029de2 */
        /*17d8*/                   MUFU.RSQ64H R9, R7;                                         /* 0xc80000001c725c00 */
        /*17e0*/                   MOV32I R11, 0x3fe00000;                                     /* 0x18ff80000002dde2 */
        /*17e8*/                   DMUL R12, R8, R8;                                           /* 0x5000000020831c01 */
        /*17f0*/                   DFMA R12, -R6, R12, c[0x2][0x0];                            /* 0x2018880000631e01 */
        /*17f8*/                   DFMA R10, R12, 0.375, R10;                                  /* 0x2014cff600c29c01 */
                                                                                               /* 0x2202929282029297 */
        /*1808*/                   DMUL R12, R8, R12;                                          /* 0x5000000030831c01 */
        /*1810*/                   DFMA R10, R10, R12, R8;                                     /* 0x2010000030a29c01 */
        /*1818*/                   DMUL R8, R6, R10;                                           /* 0x5000000028621c01 */
        /*1820*/                   IADD32I R11, R11, -0x100000;                                /* 0x0bffc00000b2dc02 */
        /*1828*/                   DFMA R6, -R8, R8, R6;                                       /* 0x200c000020819e01 */
        /*1830*/                   DFMA R8, R10, R6, R8;                                       /* 0x2010000018a21c01 */
        /*1838*/                   IADD32I R9, R9, -0x3500000;                                 /* 0x0bf2c00000925c02 */
                                                                                               /* 0x2002e20042b202e7 */
        /*1848*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1850*/                   DADD R8, R6, R6;                                            /* 0x4800000018621c01 */
        /*1858*/                   BRK;                                                        /* 0xa800000000001de7 */
        /*1860*/                   MOV R30, R8;                                                /* 0x2800000020079de4 */
        /*1868*/                   MOV R31, R9;                                                /* 0x280000002407dde4 */
        /*1870*/                   RET;                                                        /* 0x9000000000001de7 */
        /*1878*/                   BRA 0x1878;                                                 /* 0x4003ffffe0001de7 */
		.....................................................................................................................


		Function : _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE_clEvEUlPT_DpT0_E_JP4BodyEEEvS4_mS7_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                      /* 0x22c2b20282823307 */
        /*0008*/                   MOV R1, c[0x0][0x44];                              /* 0x2800400110005de4 */
        /*0010*/                   S2R R0, SR_CTAID.X;                                /* 0x2c00000094001c04 */
        /*0018*/                   S2R R3, SR_TID.X;                                  /* 0x2c0000008400dc04 */
        /*0020*/                   IMAD R0, R0, c[0x0][0x28], R3;                     /* 0x20064000a0001ca3 */
        /*0028*/                   ISUB RZ.CC, R0, c[0x0][0x140];                     /* 0x48014005000fdd03 */
        /*0030*/                   SHR R2, R0, 0x1f;                                  /* 0x5800c0007c009c23 */
        /*0038*/                   ISETP.GE.U32.X.AND P0, PT, R2, c[0x0][0x144], PT;  /* 0x1b0e40051021dc43 */
                                                                                      /* 0x2200423042c042e7 */
        /*0048*/               @P0 EXIT;                                              /* 0x80000000000001e7 */
        /*0050*/                   IADD R4.CC, R0, c[0x0][0x150];                     /* 0x4801400540011c03 */
        /*0058*/                   MOV32I R0, 0x3;                                    /* 0x180000000c001de2 */
        /*0060*/                   IADD.X R2, R2, c[0x0][0x154];                      /* 0x4800400550209c43 */
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
		....................................................................................................................


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


		Function : _Z22codengen_simple_updateP4Bodyd
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
        /*0128*/                   CAL 0x268;                                                  /* 0x50000004e0011c07 */
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
                                                                                               /* 0x22e2e04292b2b047 */
        /*0208*/                   MOV R14, R16;                                               /* 0x2800000040039de4 */
        /*0210*/                   MOV R15, R17;                                               /* 0x280000004403dde4 */
        /*0218*/                   CAL 0x268;                                                  /* 0x5000000120011c07 */
        /*0220*/                   DADD R4, R4, R22;                                           /* 0x4800000058411c01 */
        /*0228*/                   ST.E.64 [R2+0x3e810], R4;                                   /* 0x94000fa040211ca5 */
        /*0230*/                   DFMA R6, R12, c[0x0][0x148], R6;                            /* 0x200c400520c19c01 */
        /*0238*/                   LD.E.64 R10, [R2+0x3e810];                                  /* 0x84000fa040229ca5 */
                                                                                               /* 0x22028042e04293f7 */
        /*0248*/                   ST.E.64 [R2+0xfa10], R6;                                    /* 0x940003e840219ca5 */
        /*0250*/                   DFMA R8, R10, c[0x0][0x148], R8;                            /* 0x2010400520a21c01 */
        /*0258*/                   ST.E.64 [R2+0x1f410], R8;                                   /* 0x940007d040221ca5 */
        /*0260*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*0268*/                   LOP32I.AND R0, R11, 0x40000000;                             /* 0x3900000000b01c02 */
        /*0270*/                   LOP32I.AND R26, R15, 0x7f800000;                            /* 0x39fe000000f69c02 */
        /*0278*/                   ISETP.LT.U32.AND P0, PT, R0, c[0x2][0x10], PT;              /* 0x188e48004001dc03 */
                                                                                               /* 0x22c042028042b047 */
        /*0288*/                   MOV32I R0, 0x1ff00000;                                      /* 0x187fc00000001de2 */
        /*0290*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*0298*/                   SEL R17, R0, c[0x2][0x14], !P0;                             /* 0x2010480050045c04 */
        /*02a0*/                   MOV32I R20, 0x1;                                            /* 0x1800000004051de2 */
        /*02a8*/                   DMUL R18, R10, R16;                                         /* 0x5000000040a49c01 */
        /*02b0*/                   ISETP.LT.U32.AND P0, PT, R26, c[0x2][0x18], PT;             /* 0x188e480061a1dc03 */
        /*02b8*/                   MOV R24, RZ;                                                /* 0x28000000fc061de4 */
                                                                                               /* 0x2292925232928047 */
        /*02c8*/                   MUFU.RCP64H R21, R19;                                       /* 0xc800000019355c00 */
        /*02d0*/                   SEL R25, R0, c[0x2][0x14], !P0;                             /* 0x2010480050065c04 */
        /*02d8*/                   DFMA R22, -R18, R20, c[0x2][0x0];                           /* 0x2028880001259e01 */
        /*02e0*/                   DFMA R30, R22, R22, R22;                                    /* 0x202c000059679c01 */
        /*02e8*/                   DMUL R22, R14, R24;                                         /* 0x5000000060e59c01 */
        /*02f0*/                   DFMA R20, R20, R30, R20;                                    /* 0x2028000079451c01 */
        /*02f8*/                   DMUL R24, R22, R20;                                         /* 0x5000000051661c01 */
                                                                                               /* 0x22823042b2d29297 */
        /*0308*/                   DFMA R18, -R18, R24, R22;                                   /* 0x202c000061249e01 */
        /*0310*/                   DFMA R22, R20, R18, R24;                                    /* 0x2030000049459c01 */
        /*0318*/                   DSETP.LEU.AND P0, PT, |R22|, RZ, PT;                        /* 0x1d8e0000fd61dc81 */
        /*0320*/               @P0 BRA 0x4d8;                                                  /* 0x40000006c00001e7 */
        /*0328*/                   ISETP.GT.U32.AND P0, PT, R26, c[0x2][0x1c], PT;             /* 0x1a0e480071a1dc03 */
        /*0330*/                   MOV R18, RZ;                                                /* 0x28000000fc049de4 */
        /*0338*/                   DMUL R20, R16, R22;                                         /* 0x5000000059051c01 */
                                                                                               /* 0x22d2925232523287 */
        /*0348*/                   SEL R19, R0, c[0x2][0x14], !P0;                             /* 0x201048005004dc04 */
        /*0350*/                   DMUL R22, R22, R18;                                         /* 0x5000000049659c01 */
        /*0358*/                   DMUL R20, R18, R20;                                         /* 0x5000000051251c01 */
        /*0360*/                   DMUL R16, R16, R22;                                         /* 0x5000000059041c01 */
        /*0368*/                   DFMA R18, R10, R20, -R14;                                   /* 0x201c000050a49d01 */
        /*0370*/                   DFMA R22, R10, R16, -R14;                                   /* 0x201c000040a59d01 */
        /*0378*/                   DSETP.GT.AND P0, PT, |R18|, |R22|, PT;                      /* 0x1a0e00005921dcc1 */
                                                                                               /* 0x22220042b2c27207 */
        /*0388*/                   SEL R17, R17, R21, P0;                                      /* 0x2000000055145c04 */
        /*0390*/                   SEL R16, R16, R20, P0;                                      /* 0x2000000051041c04 */
        /*0398*/                   FSETP.GTU.AND P0, PT, |R17|, 1.469367938527859385e-39, PT;  /* 0x260ec0040111dc80 */
        /*03a0*/               @P0 BRA 0x550;                                                  /* 0x40000006a00001e7 */
        /*03a8*/                   FSETP.LT.AND P0, PT, |R15|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000f1dc80 */
        /*03b0*/                   LOP32I.AND R0, R16, 0xfffffffe;                             /* 0x3bfffffff9001c02 */
        /*03b8*/                   MOV32I R21, 0x3ff00000;                                     /* 0x18ffc00000055de2 */
                                                                                               /* 0x2202023280428047 */
        /*03c8*/                   MOV R20, RZ;                                                /* 0x28000000fc051de4 */
        /*03d0*/                   LOP32I.OR R22, R16, 0x1;                                    /* 0x3800000005059c42 */
        /*03d8*/                   SEL R21, R21, c[0x2][0x20], !P0;                            /* 0x2010480081555c04 */
        /*03e0*/                   MOV R16, R0;                                                /* 0x2800000000041de4 */
        /*03e8*/                   DMUL R10, R10, R20;                                         /* 0x5000000050a29c01 */
        /*03f0*/                   DMUL R14, R14, R20;                                         /* 0x5000000050e39c01 */
        /*03f8*/                   MOV R20, R22;                                               /* 0x2800000058051de4 */
                                                                                               /* 0x22c04282d2923277 */
        /*0408*/                   MOV R21, R17;                                               /* 0x2800000044055de4 */
        /*0410*/                   DFMA R18, R16, R10, -R14;                                   /* 0x201c000029049d01 */
        /*0418*/                   DFMA R20, R20, R10, -R14;                                   /* 0x201c000029451d01 */
        /*0420*/                   DSETP.GT.AND P0, PT, |R18|, |R20|, PT;                      /* 0x1a0e00005121dcc1 */
        /*0428*/                   SEL R23, R22, R0, P0;                                       /* 0x200000000165dc04 */
        /*0430*/                   IADD32I R0.CC, R23, 0x1;                                    /* 0x0c00000005701c02 */
        /*0438*/                   LOP32I.AND R18, R23, 0x1;                                   /* 0x3800000005749c02 */
                                                                                               /* 0x2262020042c20047 */
        /*0448*/                   ISETP.EQ.U32.AND P0, PT, R18, 0x1, PT;                      /* 0x190ec0000521dc03 */
        /*0450*/                   IADD.X R16, RZ, R17;                                        /* 0x4800000047f41c43 */
        /*0458*/                   IADD32I R20.CC, R23, -0x1;                                  /* 0x0ffffffffd751c02 */
        /*0460*/                   SEL R0, R23, R0, !P0;                                       /* 0x2010000001701c04 */
        /*0468*/                   IADD32I.X R21, R17, -0x1;                                   /* 0x0bfffffffd155c42 */
        /*0470*/                   SEL R19, R17, R16, !P0;                                     /* 0x201000004114dc04 */
        /*0478*/                   SEL R20, R20, R23, !P0;                                     /* 0x201000005d451c04 */
                                                                                               /* 0x2202d29232820207 */
        /*0488*/                   SEL R21, R21, R17, !P0;                                     /* 0x2010000045555c04 */
        /*0490*/                   MOV R16, R0;                                                /* 0x2800000000041de4 */
        /*0498*/                   MOV R17, R19;                                               /* 0x280000004c045de4 */
        /*04a0*/                   DFMA R16, R10, R16, -R14;                                   /* 0x201c000040a41d01 */
        /*04a8*/                   DFMA R10, R10, R20, -R14;                                   /* 0x201c000050a29d01 */
        /*04b0*/                   DSETP.GT.AND P0, PT, |R16|, |R10|, PT;                      /* 0x1a0e00002901dcc1 */
        /*04b8*/                   SEL R16, R20, R0, P0;                                       /* 0x2000000001441c04 */
                                                                                               /* 0x22d28042b2d2e047 */
        /*04c8*/                   SEL R17, R21, R19, P0;                                      /* 0x200000004d545c04 */
        /*04d0*/                   BRA 0x550;                                                  /* 0x40000001e0001de7 */
        /*04d8*/                   DSETP.EQ.AND P0, PT, R22, RZ, PT;                           /* 0x190e0000fd61dc01 */
        /*04e0*/               @P0 BRA 0x540;                                                  /* 0x40000001600001e7 */
        /*04e8*/                   MUFU.RCP64H R17, R11;                                       /* 0xc800000018b45c00 */
        /*04f0*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*04f8*/                   DSETP.GT.AND P1, PT, |R16|, RZ, PT;                         /* 0x1a0e0000fd03dc81 */
                                                                                               /* 0x22e20282027202d7 */
        /*0508*/              @!P1 DSETP.NEU.AND P0, PT, |R10|, +INF , PT;                     /* 0x1e8edffc00a1e481 */
        /*0510*/              @!P1 SEL R0, R10, R16, P0;                                       /* 0x2000000040a02404 */
        /*0518*/              @!P1 SEL R11, R11, R17, P0;                                      /* 0x2000000044b2e404 */
        /*0520*/              @!P1 MOV R16, R0;                                                /* 0x28000000000425e4 */
        /*0528*/              @!P1 MOV R17, R11;                                               /* 0x280000002c0465e4 */
        /*0530*/                   DMUL R16, R14, R16;                                         /* 0x5000000040e41c01 */
        /*0538*/                   BRA 0x550;                                                  /* 0x4000000040001de7 */
                                                                                               /* 0x20000002e20042b7 */
        /*0548*/                   DMUL R16, R14, R10;                                         /* 0x5000000028e41c01 */
        /*0550*/                   MOV R22, R16;                                               /* 0x2800000040059de4 */
        /*0558*/                   MOV R23, R17;                                               /* 0x280000004405dde4 */
        /*0560*/                   RET;                                                        /* 0x9000000000001de7 */
        /*0568*/                   BRA 0x568;                                                  /* 0x4003ffffe0001de7 */
        /*0570*/                   NOP;                                                        /* 0x4000000000001de4 */
        /*0578*/                   NOP;                                                        /* 0x4000000000001de4 */
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
                                                                                               /* 0x2202e2c2c28042b7 */
        /*0008*/                   MOV R1, c[0x0][0x44];                                       /* 0x2800400110005de4 */
        /*0010*/                   MOV R15, c[0x0][0x140];                                     /* 0x280040050003dde4 */
        /*0018*/                   MOV R5, c[0x0][0x144];                                      /* 0x2800400510015de4 */
        /*0020*/                   ISUB RZ.CC, R15, c[0x0][0x148];                             /* 0x4801400520ffdd03 */
        /*0028*/                   ISETP.EQ.X.AND P0, PT, R5, c[0x0][0x14c], PT;               /* 0x190e40053051dc63 */
        /*0030*/               @P0 EXIT;                                                       /* 0x80000000000001e7 */
        /*0038*/                   MOV32I R0, 0x3;                                             /* 0x180000000c001de2 */
                                                                                               /* 0x2282204200427047 */
        /*0048*/                   MOV32I R8, 0x0;                                             /* 0x1800000000021de2 */
        /*0050*/                   SHL R16, R15, -0x1d;                                        /* 0x6000ffff8cf41c03 */
        /*0058*/                   IADD32I RZ.CC, R0, -0x20;                                   /* 0x0fffffff800fdc02 */
        /*0060*/                   SHR.U32 R6, R15, 0x1d;                                      /* 0x5800c00074f19c03 */
        /*0068*/                   MOV32I R17, 0x0;                                            /* 0x1800000000045de2 */
        /*0070*/                   MOV R9, c[0x0][0x148];                                      /* 0x2800400520025de4 */
        /*0078*/                   SHL R15, R15, 0x3;                                          /* 0x6000c0000cf3dc03 */
                                                                                               /* 0x2202304280423047 */
        /*0088*/                   CSETP.SFT.AND P0, PT, CC, PT;                               /* 0x040e00000151dc04 */
        /*0090*/                   MOV R4, c[0x0][0x14c];                                      /* 0x2800400530011de4 */
        /*0098*/                   IADD32I R2.CC, R8, 0xfa10;                                  /* 0x0c0003e840809c02 */
        /*00a0*/                   SHL R5, R5, 0x3;                                            /* 0x6000c0000c515c03 */
        /*00a8*/                   SHL R14, R9, 0x3;                                           /* 0x6000c0000c939c03 */
        /*00b0*/               @P0 IADD R16, R5, R6;                                           /* 0x4800000018540003 */
        /*00b8*/                   IADD.X R3, RZ, R17;                                         /* 0x4800000047f0dc43 */
                                                                                               /* 0x2232320232323207 */
        /*00c8*/                   IADD R7.CC, R15, R2;                                        /* 0x4801000008f1dc03 */
        /*00d0*/                   SHL R4, R4, 0x3;                                            /* 0x6000c0000c411c03 */
        /*00d8*/                   SHL R0, R9, -0x1d;                                          /* 0x6000ffff8c901c03 */
        /*00e0*/                   SHR.U32 R5, R9, 0x1d;                                       /* 0x5800c00074915c03 */
        /*00e8*/                   IADD.X R6, R16, R3;                                         /* 0x480000000d019c43 */
        /*00f0*/                   IADD R2.CC, R2, R14;                                        /* 0x4801000038209c03 */
        /*00f8*/               @P0 IADD R0, R4, R5;                                            /* 0x4800000014400003 */
                                                                                               /* 0x23f2323280428047 */
        /*0108*/                   MOV R4, R7;                                                 /* 0x280000001c011de4 */
        /*0110*/                   MOV R5, R6;                                                 /* 0x2800000018015de4 */
        /*0118*/                   IADD.X R3, R3, R0;                                          /* 0x480000000030dc43 */
        /*0120*/                   LD.E.64 R12, [R4+0xfa00];                                   /* 0x840003e800431ca5 */
        /*0128*/                   LD.E.64 R10, [R2+0xfa00];                                   /* 0x840003e800229ca5 */
        /*0130*/                   LD.E.64 R24, [R4];                                          /* 0x8400000000461ca5 */
        /*0138*/                   LD.E.64 R6, [R2];                                           /* 0x8400000000219ca5 */
                                                                                               /* 0x2292804292923277 */
        /*0148*/                   DADD R12, -R12, R10;                                        /* 0x4800000028c31e01 */
        /*0150*/                   DADD R24, -R24, R6;                                         /* 0x4800000019861e01 */
        /*0158*/                   DMUL R6, R12, R12;                                          /* 0x5000000030c19c01 */
        /*0160*/                   DFMA R6, R24, R24, R6;                                      /* 0x200c000061819c01 */
        /*0168*/                   IADD32I R18, R7, -0x3500000;                                /* 0x0bf2c00000749c02 */
        /*0170*/                   MUFU.RSQ64H R19, R7;                                        /* 0xc80000001c74dc00 */
        /*0178*/                   DMUL R10, R18, R18;                                         /* 0x5000000049229c01 */
                                                                                               /* 0x2282029232804207 */
        /*0188*/                   DFMA R20, -R6, R10, c[0x2][0x0];                            /* 0x2014880000651e01 */
        /*0190*/                   MOV32I R10, 0x0;                                            /* 0x1800000000029de2 */
        /*0198*/                   MOV32I R11, 0x3fe00000;                                     /* 0x18ff80000002dde2 */
        /*01a0*/                   DFMA R4, R20, 0.375, R10;                                   /* 0x2014cff601411c01 */
        /*01a8*/                   DMUL R22, R18, R20;                                         /* 0x5000000051259c01 */
        /*01b0*/                   DFMA R4, R4, R22, R18;                                      /* 0x2024000058411c01 */
        /*01b8*/                   IADD32I R19.CC, R8, 0x10;                                   /* 0x0c0000004084dc02 */
                                                                                               /* 0x2202926042220047 */
        /*01c8*/                   ISETP.LT.U32.AND P0, PT, R18, c[0x2][0x8], PT;              /* 0x188e48002121dc03 */
        /*01d0*/                   IADD32I R21, R5, -0x100000;                                 /* 0x0bffc00000555c02 */
        /*01d8*/                   DMUL R2, R6, R4;                                            /* 0x5000000010609c01 */
        /*01e0*/                   MOV R20, R4;                                                /* 0x2800000010051de4 */
        /*01e8*/                   IADD.X R17, RZ, R17;                                        /* 0x4800000047f45c43 */
        /*01f0*/                   DFMA R8, -R2, R2, R6;                                       /* 0x200c000008221e01 */
        /*01f8*/                   DFMA R10, R8, R20, R2;                                      /* 0x2004000050829c01 */
                                                                                               /* 0x22b2b200420042b7 */
        /*0208*/               @P0 BRA 0x240;                                                  /* 0x40000000c00001e7 */
        /*0210*/                   MOV R10, R4;                                                /* 0x2800000010029de4 */
        /*0218*/                   MOV R4, R8;                                                 /* 0x2800000020011de4 */
        /*0220*/                   MOV R5, R9;                                                 /* 0x2800000024015de4 */
        /*0228*/                   MOV R8, R2;                                                 /* 0x2800000008021de4 */
        /*0230*/                   MOV R9, R3;                                                 /* 0x280000000c025de4 */
        /*0238*/                   CAL 0x7b8;                                                  /* 0x50000015e0011c07 */
                                                                                               /* 0x22824042704202c7 */
        /*0248*/                   IADD R8.CC, R15, R19;                                       /* 0x480100004cf21c03 */
        /*0250*/                   IADD.X R9, R16, R17;                                        /* 0x4800000045025c43 */
        /*0258*/                   IADD R18.CC, R19, R14;                                      /* 0x4801000039349c03 */
        /*0260*/                   MOV32I R14, 0x1;                                            /* 0x1800000004039de2 */
        /*0268*/                   LD.E.64 R4, [R8];                                           /* 0x8400000000811ca5 */
        /*0270*/                   DFMA R2, R10, R10, c[0x2][0x18];                            /* 0x2014880060a09c01 */
        /*0278*/                   IADD.X R19, R17, R0;                                        /* 0x480000000114dc43 */
                                                                                               /* 0x2292d3f292928047 */
        /*0288*/                   LD.E.64 R6, [R18];                                          /* 0x8400000001219ca5 */
        /*0290*/                   MUFU.RCP64H R15, R3;                                        /* 0xc80000001833dc00 */
        /*0298*/                   DFMA R16, -R2, R14, c[0x2][0x0];                            /* 0x201c880000241e01 */
        /*02a0*/                   DFMA R16, R16, R16, R16;                                    /* 0x2020000041041c01 */
        /*02a8*/                   DFMA R16, R14, R16, R14;                                    /* 0x201c000040e41c01 */
        /*02b0*/                   DMUL R4, R4, c[0x2][0x10];                                  /* 0x5000480040411c01 */
        /*02b8*/                   DMUL R14, R4, R6;                                           /* 0x5000000018439c01 */
                                                                                               /* 0x22b2c28282029297 */
        /*02c8*/                   DMUL R4, R14, R16;                                          /* 0x5000000040e11c01 */
        /*02d0*/                   DFMA R6, -R2, R4, R14;                                      /* 0x201c000010219e01 */
        /*02d8*/                   DFMA R6, R16, R6, R4;                                       /* 0x2008000019019c01 */
        /*02e0*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0xc], PT;                 /* 0x160f480030ffdc80 */
        /*02e8*/                   FFMA R0, RZ, R3, R7;                                        /* 0x300e00000ff01c00 */
        /*02f0*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0x20], PT;                /* 0x220e48008001dc80 */
        /*02f8*/               @P0 BRA CC.NEU, 0x330;                                          /* 0x40000000c00001a7 */
                                                                                               /* 0x228042b042b2b047 */
        /*0308*/                   MOV R4, R2;                                                 /* 0x2800000008011de4 */
        /*0310*/                   MOV R5, R3;                                                 /* 0x280000000c015de4 */
        /*0318*/                   CAL 0x4e0;                                                  /* 0x5000000700011c07 */
        /*0320*/                   MOV R6, R2;                                                 /* 0x2800000008019de4 */
        /*0328*/                   MOV R7, R3;                                                 /* 0x280000000c01dde4 */
        /*0330*/                   MUFU.RCP64H R5, R11;                                        /* 0xc800000018b15c00 */
        /*0338*/                   MOV32I R4, 0x1;                                             /* 0x1800000004011de2 */
                                                                                               /* 0x2202929292523297 */
        /*0348*/                   DFMA R2, -R10, R4, c[0x2][0x0];                             /* 0x2008880000a09e01 */
        /*0350*/                   DFMA R2, R2, R2, R2;                                        /* 0x2004000008209c01 */
        /*0358*/                   DMUL R24, R24, R6;                                          /* 0x5000000019861c01 */
        /*0360*/                   DFMA R4, R4, R2, R4;                                        /* 0x2008000008411c01 */
        /*0368*/                   DMUL R14, R24, R4;                                          /* 0x5000000011839c01 */
        /*0370*/                   DFMA R2, -R10, R14, R24;                                    /* 0x2030000038a09e01 */
        /*0378*/                   DFMA R4, R4, R2, R14;                                       /* 0x201c000008411c01 */
                                                                                               /* 0x2202b042b2028287 */
        /*0388*/                   FSET.GTU.AND RZ.CC, |R25|, c[0x2][0xc], PT;                 /* 0x160f4800319fdc80 */
        /*0390*/                   FFMA R0, RZ, R11, R5;                                       /* 0x300a00002ff01c00 */
        /*0398*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0x20], PT;                /* 0x220e48008001dc80 */
        /*03a0*/                   MOV R0, R10;                                                /* 0x2800000028001de4 */
        /*03a8*/                   MOV R2, R11;                                                /* 0x280000002c009de4 */
        /*03b0*/               @P0 BRA CC.NEU, 0x3f8;                                          /* 0x40000001000001a7 */
        /*03b8*/                   MOV R14, R24;                                               /* 0x2800000060039de4 */
                                                                                               /* 0x2232b042b2b20047 */
        /*03c8*/                   MOV R15, R25;                                               /* 0x280000006403dde4 */
        /*03d0*/                   MOV R4, R0;                                                 /* 0x2800000000011de4 */
        /*03d8*/                   MOV R5, R2;                                                 /* 0x2800000008015de4 */
        /*03e0*/                   CAL 0x4e0;                                                  /* 0x50000003e0011c07 */
        /*03e8*/                   MOV R4, R2;                                                 /* 0x2800000008011de4 */
        /*03f0*/                   MOV R5, R3;                                                 /* 0x280000000c015de4 */
        /*03f8*/                   LD.E.64 R2, [R8+0x4e200];                                   /* 0x8400138800809ca5 */
                                                                                               /* 0x2292523292820047 */
        /*0408*/                   LD.E.64 R22, [R8+0x5dc00];                                  /* 0x8400177000859ca5 */
        /*0410*/                   MUFU.RCP64H R17, R11;                                       /* 0xc800000018b45c00 */
        /*0418*/                   MOV32I R16, 0x1;                                            /* 0x1800000004041de2 */
        /*0420*/                   DFMA R14, -R10, R16, c[0x2][0x0];                           /* 0x2020880000a39e01 */
        /*0428*/                   DFMA R18, R14, R14, R14;                                    /* 0x201c000038e49c01 */
        /*0430*/                   DMUL R14, R12, R6;                                          /* 0x5000000018c39c01 */
        /*0438*/                   DFMA R16, R16, R18, R16;                                    /* 0x2020000049041c01 */
                                                                                               /* 0x2293f28282029297 */
        /*0448*/                   DMUL R12, R16, R14;                                         /* 0x5000000039031c01 */
        /*0450*/                   DFMA R6, -R10, R12, R14;                                    /* 0x201c000030a19e01 */
        /*0458*/                   DFMA R12, R16, R6, R12;                                     /* 0x2018000019031c01 */
        /*0460*/                   FSET.GTU.AND RZ.CC, |R15|, c[0x2][0xc], PT;                 /* 0x160f480030ffdc80 */
        /*0468*/                   FFMA R0, RZ, R11, R13;                                      /* 0x301a00002ff01c00 */
        /*0470*/                   FSETP.GT.AND P0, PT, |R0|, c[0x2][0x20], PT;                /* 0x220e48008001dc80 */
        /*0478*/                   DADD R2, R4, R2;                                            /* 0x4800000008409c01 */
                                                                                               /* 0x22b042b2b0420047 */
        /*0488*/                   ST.E.64 [R8+0x4e200], R2;                                   /* 0x9400138800809ca5 */
        /*0490*/                   MOV R4, R10;                                                /* 0x2800000028011de4 */
        /*0498*/                   MOV R5, R11;                                                /* 0x280000002c015de4 */
        /*04a0*/               @P0 BRA CC.NEU, 0x4c0;                                          /* 0x40000000600001a7 */
        /*04a8*/                   CAL 0x4e0;                                                  /* 0x50000000c0011c07 */
        /*04b0*/                   MOV R12, R2;                                                /* 0x2800000008031de4 */
        /*04b8*/                   MOV R13, R3;                                                /* 0x280000000c035de4 */
                                                                                               /* 0x2200428042e04297 */
        /*04c8*/                   DADD R12, R22, R12;                                         /* 0x4800000031631c01 */
        /*04d0*/                   ST.E.64 [R8+0x5dc00], R12;                                  /* 0x9400177000831ca5 */
        /*04d8*/                   EXIT;                                                       /* 0x8000000000001de7 */
        /*04e0*/                   LOP32I.AND R0, R5, 0x40000000;                              /* 0x3900000000501c02 */
        /*04e8*/                   MOV R2, RZ;                                                 /* 0x28000000fc009de4 */
        /*04f0*/                   ISETP.LT.U32.AND P0, PT, R0, c[0x2][0x24], PT;              /* 0x188e48009001dc03 */
        /*04f8*/                   LOP32I.AND R26, R15, 0x7f800000;                            /* 0x39fe000000f69c02 */
                                                                                               /* 0x228280420282b047 */
        /*0508*/                   MOV32I R0, 0x1ff00000;                                      /* 0x187fc00000001de2 */
        /*0510*/                   MOV32I R18, 0x1;                                            /* 0x1800000004049de2 */
        /*0518*/                   SEL R3, R0, c[0x2][0x28], !P0;                              /* 0x20104800a000dc04 */
        /*0520*/                   DMUL R16, R4, R2;                                           /* 0x5000000008441c01 */
        /*0528*/                   ISETP.LT.U32.AND P0, PT, R26, c[0x2][0x2c], PT;             /* 0x188e4800b1a1dc03 */
        /*0530*/                   MOV R20, RZ;                                                /* 0x28000000fc051de4 */
        /*0538*/                   MUFU.RCP64H R19, R17;                                       /* 0xc80000001914dc00 */
                                                                                               /* 0x2292929252328207 */
        /*0548*/                   DFMA R28, -R16, R18, c[0x2][0x0];                           /* 0x2024880001071e01 */
        /*0550*/                   SEL R21, R0, c[0x2][0x28], !P0;                             /* 0x20104800a0055c04 */
        /*0558*/                   DFMA R28, R28, R28, R28;                                    /* 0x2038000071c71c01 */
        /*0560*/                   DMUL R20, R14, R20;                                         /* 0x5000000050e51c01 */
        /*0568*/                   DFMA R18, R18, R28, R18;                                    /* 0x2024000071249c01 */
        /*0570*/                   DMUL R30, R20, R18;                                         /* 0x5000000049479c01 */
        /*0578*/                   DFMA R16, -R16, R30, R20;                                   /* 0x2028000079041e01 */
                                                                                               /* 0x2282823042b2d297 */
        /*0588*/                   DFMA R20, R18, R16, R30;                                    /* 0x203c000041251c01 */
        /*0590*/                   DSETP.LEU.AND P0, PT, |R20|, RZ, PT;                        /* 0x1d8e0000fd41dc81 */
        /*0598*/               @P0 BRA 0x738;                                                  /* 0x40000006600001e7 */
        /*05a0*/                   ISETP.GT.U32.AND P0, PT, R26, c[0x2][0x30], PT;             /* 0x1a0e4800c1a1dc03 */
        /*05a8*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*05b0*/                   DMUL R18, R2, R20;                                          /* 0x5000000050249c01 */
        /*05b8*/                   SEL R17, R0, c[0x2][0x28], !P0;                             /* 0x20104800a0045c04 */
                                                                                               /* 0x2202d29252325237 */
        /*05c8*/                   DMUL R20, R20, R16;                                         /* 0x5000000041451c01 */
        /*05d0*/                   DMUL R16, R16, R18;                                         /* 0x5000000049041c01 */
        /*05d8*/                   DMUL R20, R2, R20;                                          /* 0x5000000050251c01 */
        /*05e0*/                   DFMA R2, R4, R16, -R14;                                     /* 0x201c000040409d01 */
        /*05e8*/                   DFMA R28, R4, R20, -R14;                                    /* 0x201c000050471d01 */
        /*05f0*/                   DSETP.GT.AND P0, PT, |R2|, |R28|, PT;                       /* 0x1a0e00007021dcc1 */
        /*05f8*/                   SEL R3, R21, R17, P0;                                       /* 0x200000004550dc04 */
                                                                                               /* 0x2220420042e2c277 */
        /*0608*/                   SEL R2, R20, R16, P0;                                       /* 0x2000000041409c04 */
        /*0610*/                   FSETP.GTU.AND P0, PT, |R3|, 1.469367938527859385e-39, PT;   /* 0x260ec0040031dc80 */
        /*0618*/               @P0 RET;                                                        /* 0x90000000000001e7 */
        /*0620*/                   FSETP.LT.AND P0, PT, |R15|, 1.5046327690525280102e-36, PT;  /* 0x208ec10000f1dc80 */
        /*0628*/                   LOP32I.AND R0, R2, 0xfffffffe;                              /* 0x3bfffffff8201c02 */
        /*0630*/                   MOV32I R17, 0x3ff00000;                                     /* 0x18ffc00000045de2 */
        /*0638*/                   MOV R21, R3;                                                /* 0x280000000c055de4 */
                                                                                               /* 0x2232923280428047 */
        /*0648*/                   MOV R16, RZ;                                                /* 0x28000000fc041de4 */
        /*0650*/                   LOP32I.OR R20, R2, 0x1;                                     /* 0x3800000004251c42 */
        /*0658*/                   SEL R17, R17, c[0x2][0x34], !P0;                            /* 0x20104800d1145c04 */
        /*0660*/                   MOV R2, R0;                                                 /* 0x2800000000009de4 */
        /*0668*/                   DMUL R4, R4, R16;                                           /* 0x5000000040411c01 */
        /*0670*/                   DMUL R14, R14, R16;                                         /* 0x5000000040e39c01 */
        /*0678*/                   DFMA R18, R20, R4, -R14;                                    /* 0x201c000011449d01 */
                                                                                               /* 0x228232804282d297 */
        /*0688*/                   DFMA R16, R2, R4, -R14;                                     /* 0x201c000010241d01 */
        /*0690*/                   DSETP.GT.AND P0, PT, |R16|, |R18|, PT;                      /* 0x1a0e00004901dcc1 */
        /*0698*/                   SEL R20, R20, R0, P0;                                       /* 0x2000000001451c04 */
        /*06a0*/                   IADD32I R0.CC, R20, 0x1;                                    /* 0x0c00000005401c02 */
        /*06a8*/                   LOP32I.AND R2, R20, 0x1;                                    /* 0x3800000005409c02 */
        /*06b0*/                   ISETP.EQ.U32.AND P0, PT, R2, 0x1, PT;                       /* 0x190ec0000421dc03 */
        /*06b8*/                   IADD.X R16, RZ, R3;                                         /* 0x480000000ff41c43 */
                                                                                               /* 0x2202720042b20047 */
        /*06c8*/                   IADD32I R18.CC, R20, -0x1;                                  /* 0x0ffffffffd449c02 */
        /*06d0*/                   SEL R0, R20, R0, !P0;                                       /* 0x2010000001401c04 */
        /*06d8*/                   SEL R17, R3, R16, !P0;                                      /* 0x2010000040345c04 */
        /*06e0*/                   IADD32I.X R2, R3, -0x1;                                     /* 0x0bfffffffc309c42 */
        /*06e8*/                   SEL R18, R18, R20, !P0;                                     /* 0x2010000051249c04 */
        /*06f0*/                   MOV R16, R0;                                                /* 0x2800000000041de4 */
        /*06f8*/                   SEL R19, R2, R3, !P0;                                       /* 0x201000000c24dc04 */
                                                                                               /* 0x22d2e04202d29277 */
        /*0708*/                   DFMA R2, R4, R16, -R14;                                     /* 0x201c000040409d01 */
        /*0710*/                   DFMA R4, R4, R18, -R14;                                     /* 0x201c000048411d01 */
        /*0718*/                   DSETP.GT.AND P0, PT, |R2|, |R4|, PT;                        /* 0x1a0e00001021dcc1 */
        /*0720*/                   SEL R2, R18, R0, P0;                                        /* 0x2000000001209c04 */
        /*0728*/                   SEL R3, R19, R17, P0;                                       /* 0x200000004530dc04 */
        /*0730*/                   RET;                                                        /* 0x9000000000001de7 */
        /*0738*/                   DSETP.EQ.AND P0, PT, R20, RZ, PT;                           /* 0x190e0000fd41dc01 */
                                                                                               /* 0x227202d2d28042b7 */
        /*0748*/               @P0 BRA 0x7a8;                                                  /* 0x40000001600001e7 */
        /*0750*/                   MUFU.RCP64H R3, R5;                                         /* 0xc80000001850dc00 */
        /*0758*/                   MOV R2, RZ;                                                 /* 0x28000000fc009de4 */
        /*0760*/                   DSETP.GT.AND P1, PT, |R2|, RZ, PT;                          /* 0x1a0e0000fc23dc81 */
        /*0768*/              @!P1 DSETP.NEU.AND P0, PT, |R4|, +INF , PT;                      /* 0x1e8edffc0041e481 */
        /*0770*/              @!P1 SEL R0, R4, R2, P0;                                         /* 0x2000000008402404 */
        /*0778*/              @!P1 SEL R4, R5, R3, P0;                                         /* 0x200000000c512404 */
                                                                                               /* 0x2202e202e2028207 */
        /*0788*/              @!P1 MOV R2, R0;                                                 /* 0x280000000000a5e4 */
        /*0790*/              @!P1 MOV R3, R4;                                                 /* 0x280000001000e5e4 */
        /*0798*/                   DMUL R2, R14, R2;                                           /* 0x5000000008e09c01 */
        /*07a0*/                   RET;                                                        /* 0x9000000000001de7 */
        /*07a8*/                   DMUL R2, R14, R4;                                           /* 0x5000000010e09c01 */
        /*07b0*/                   RET;                                                        /* 0x9000000000001de7 */
        /*07b8*/                   ISETP.LT.U32.AND P0, PT, R18, c[0x2][0x38], PT;             /* 0x188e4800e121dc03 */
                                                                                               /* 0x22c292b2a0420047 */
        /*07c8*/                   MOV R2, R6;                                                 /* 0x2800000018009de4 */
        /*07d0*/                   MOV R3, R7;                                                 /* 0x280000001c00dde4 */
        /*07d8*/                   MOV R6, R10;                                                /* 0x2800000028019de4 */
        /*07e0*/                   MOV R7, R21;                                                /* 0x280000005401dde4 */
        /*07e8*/               @P0 BRA 0x838;                                                  /* 0x40000001200001e7 */
        /*07f0*/                   DFMA.RM R4, R4, R6, R8;                                     /* 0x2090000018411c01 */
        /*07f8*/                   IADD32I R6.CC, R4, 0x1;                                     /* 0x0c00000004419c02 */
                                                                                               /* 0x22d2e04202d29287 */
        /*0808*/                   IADD.X R7, RZ, R5;                                          /* 0x4800000017f1dc43 */
        /*0810*/                   DFMA.RP R2, -R4, R6, R2;                                    /* 0x2104000018409e01 */
        /*0818*/                   DSETP.GT.AND P0, PT, R2, RZ, PT;                            /* 0x1a0e0000fc21dc01 */
        /*0820*/                   SEL R4, R6, R4, P0;                                         /* 0x2000000010611c04 */
        /*0828*/                   SEL R5, R7, R5, P0;                                         /* 0x2000000014715c04 */
        /*0830*/                   BRA 0x920;                                                  /* 0x40000003a0001de7 */
        /*0838*/                   DSETP.EQU.AND P0, PT, R2, RZ, PT;                           /* 0x1d0e0000fc21dc01 */
                                                                                               /* 0x22b2c2b20042c2b7 */
        /*0848*/               @P0 BRA 0x918;                                                  /* 0x40000003200001e7 */
        /*0850*/                   ISETP.LT.AND P0, PT, R3, RZ, PT;                            /* 0x188e0000fc31dc23 */
        /*0858*/               @P0 MOV32I R4, 0x0;                                             /* 0x18000000000101e2 */
        /*0860*/               @P0 MOV32I R5, 0xfff80000;                                      /* 0x1bffe000000141e2 */
        /*0868*/               @P0 BRA 0x920;                                                  /* 0x40000002c00001e7 */
        /*0870*/                   ISETP.GT.AND P0, PT, R3, c[0x2][0x3c], PT;                  /* 0x1a0e4800f031dc23 */
        /*0878*/               @P0 BRA 0x918;                                                  /* 0x40000002600001e7 */
                                                                                               /* 0x2292928042804207 */
        /*0888*/                   DMUL R2, R2, 8.11296384146066816958e+31;                    /* 0x5000d1a400209c01 */
        /*0890*/                   MOV R4, RZ;                                                 /* 0x28000000fc011de4 */
        /*0898*/                   MOV32I R8, 0x0;                                             /* 0x1800000000021de2 */
        /*08a0*/                   MUFU.RSQ64H R5, R3;                                         /* 0xc80000001c315c00 */
        /*08a8*/                   MOV32I R9, 0x3fe00000;                                      /* 0x18ff800000025de2 */
        /*08b0*/                   DMUL R6, R4, R4;                                            /* 0x5000000010419c01 */
        /*08b8*/                   DFMA R6, -R2, R6, c[0x2][0x0];                              /* 0x200c880000219e01 */
                                                                                               /* 0x2292928202929237 */
        /*08c8*/                   DFMA R8, R6, 0.375, R8;                                     /* 0x2010cff600621c01 */
        /*08d0*/                   DMUL R6, R4, R6;                                            /* 0x5000000018419c01 */
        /*08d8*/                   DFMA R4, R8, R6, R4;                                        /* 0x2008000018811c01 */
        /*08e0*/                   DMUL R6, R2, R4;                                            /* 0x5000000010219c01 */
        /*08e8*/                   IADD32I R5, R5, -0x100000;                                  /* 0x0bffc00000515c02 */
        /*08f0*/                   DFMA R2, -R6, R6, R2;                                       /* 0x2004000018609e01 */
        /*08f8*/                   DFMA R4, R4, R2, R6;                                        /* 0x200c000008411c01 */
                                                                                               /* 0x2002e20042b2e047 */
        /*0908*/                   IADD32I R5, R5, -0x3500000;                                 /* 0x0bf2c00000515c02 */
        /*0910*/                   BRA 0x920;                                                  /* 0x4000000020001de7 */
        /*0918*/                   DADD R4, R2, R2;                                            /* 0x4800000008211c01 */
        /*0920*/                   MOV R10, R4;                                                /* 0x2800000010029de4 */
        /*0928*/                   MOV R11, R5;                                                /* 0x280000001402dde4 */
        /*0930*/                   RET;                                                        /* 0x9000000000001de7 */
        /*0938*/                   BRA 0x938;                                                  /* 0x4003ffffe0001de7 */
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
.global .align 8 .u64 _ZN4ikra3soa23d_previous_storage_sizeE;
.global .align 8 .u64 _ZN4ikra8executor4cuda14d_storage_sizeE;
.global .align 8 .u64 _ZN4ikra8executor4cuda19d_storage_data_headE;

.visible .entry _Z19codegen_reset_forceP4Body(
.param .u64 _Z19codegen_reset_forceP4Body_param_0
)
{
.reg .b64 %rd<7>;


ld.param.u64 %rd1, [_Z19codegen_reset_forceP4Body_param_0];
shl.b64 %rd2, %rd1, 3;
mov.u64 %rd3, __Bodydata_buffer;
cvta.global.u64 %rd4, %rd3;
add.s64 %rd5, %rd4, %rd2;
mov.u64 %rd6, 0;
st.u64 [%rd5+320016], %rd6;
st.u64 [%rd5+384016], %rd6;
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
cvta.global.u64 %rd7, %rd6;
add.s64 %rd8, %rd7, 64016;
add.s64 %rd9, %rd5, %rd8;
add.s64 %rd10, %rd7, 16;
shl.b64 %rd11, %rd1, 3;
add.s64 %rd12, %rd11, %rd8;
ld.f64 %fd1, [%rd12];
ld.f64 %fd2, [%rd9];
sub.f64 %fd3, %fd2, %fd1;
add.s64 %rd13, %rd7, 128016;
add.s64 %rd14, %rd5, %rd13;
add.s64 %rd15, %rd11, %rd13;
ld.f64 %fd4, [%rd15];
ld.f64 %fd5, [%rd14];
sub.f64 %fd6, %fd5, %fd4;
mul.f64 %fd7, %fd6, %fd6;
fma.rn.f64 %fd8, %fd3, %fd3, %fd7;
sqrt.rn.f64 %fd9, %fd8;
add.s64 %rd16, %rd11, %rd10;
ld.f64 %fd10, [%rd16];
mul.f64 %fd11, %fd10, 0d3DD257B4D002790D;
add.s64 %rd17, %rd5, %rd10;
ld.f64 %fd12, [%rd17];
mul.f64 %fd13, %fd11, %fd12;
fma.rn.f64 %fd14, %fd9, %fd9, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd15, %fd13, %fd14;
mul.f64 %fd16, %fd3, %fd15;
div.rn.f64 %fd17, %fd16, %fd9;
ld.f64 %fd18, [%rd16+320000];
add.f64 %fd19, %fd17, %fd18;
ld.f64 %fd20, [%rd16+384000];
st.f64 [%rd16+320000], %fd19;
mul.f64 %fd21, %fd6, %fd15;
div.rn.f64 %fd22, %fd21, %fd9;
add.f64 %fd23, %fd22, %fd20;
st.f64 [%rd16+384000], %fd23;

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
.reg .b64 %rd<9>;


ld.param.u64 %rd4, [_Z14codegen_updateP4Bodyd_param_0];
ld.param.f64 %fd4, [_Z14codegen_updateP4Bodyd_param_1];
shl.b64 %rd5, %rd4, 3;
mov.u64 %rd6, __Bodydata_buffer;
cvta.global.u64 %rd7, %rd6;
add.s64 %rd8, %rd7, %rd5;
ld.f64 %fd5, [%rd8+320016];
mul.f64 %fd6, %fd5, %fd4;
ld.f64 %fd7, [%rd8+16];
div.rn.f64 %fd8, %fd6, %fd7;
add.s64 %rd1, %rd8, 192016;
ld.f64 %fd9, [%rd8+192016];
ld.f64 %fd10, [%rd8+384016];
ld.f64 %fd11, [%rd8+256016];
add.f64 %fd12, %fd8, %fd9;
ld.f64 %fd13, [%rd8+64016];
ld.f64 %fd14, [%rd8+128016];
st.f64 [%rd8+192016], %fd12;
mul.f64 %fd15, %fd10, %fd4;
ld.f64 %fd16, [%rd8+16];
div.rn.f64 %fd17, %fd15, %fd16;
add.s64 %rd2, %rd8, 256016;
add.f64 %fd18, %fd17, %fd11;
st.f64 [%rd8+256016], %fd18;
fma.rn.f64 %fd19, %fd12, %fd4, %fd13;
st.f64 [%rd8+64016], %fd19;
ld.f64 %fd20, [%rd8+256016];
add.s64 %rd3, %rd8, 128016;
fma.rn.f64 %fd26, %fd20, %fd4, %fd14;
ld.f64 %fd21, [%rd8+64016];
st.f64 [%rd8+128016], %fd26;
setp.lt.f64	%p1, %fd21, 0dBFF0000000000000;
setp.gt.f64	%p2, %fd21, 0d3FF0000000000000;
or.pred %p3, %p1, %p2;
@!%p3 bra BB2_2;
bra.uni BB2_1;

BB2_1:
ld.f64 %fd22, [%rd1];
neg.f64 %fd23, %fd22;
st.f64 [%rd1], %fd23;
ld.f64 %fd26, [%rd3];

BB2_2:
setp.gt.f64	%p4, %fd26, 0d3FF0000000000000;
setp.lt.f64	%p5, %fd26, 0dBFF0000000000000;
or.pred %p6, %p5, %p4;
@!%p6 bra BB2_4;
bra.uni BB2_3;

BB2_3:
ld.f64 %fd24, [%rd2];
neg.f64 %fd25, %fd24;
st.f64 [%rd2], %fd25;

BB2_4:
ret;
}


.visible .entry _Z26codegen_reset_force_manualP4Body(
.param .u64 _Z26codegen_reset_force_manualP4Body_param_0
)
{
.reg .b64 %rd<7>;


ld.param.u64 %rd1, [_Z26codegen_reset_force_manualP4Body_param_0];
shl.b64 %rd2, %rd1, 3;
mov.u64 %rd3, __Bodydata_buffer;
cvta.global.u64 %rd4, %rd3;
add.s64 %rd5, %rd4, %rd2;
mov.u64 %rd6, 0;
st.u64 [%rd5+320016], %rd6;
st.u64 [%rd5+384016], %rd6;
ret;
}


.visible .entry _Z24codegen_reset_force_halfP4Body(
.param .u64 _Z24codegen_reset_force_halfP4Body_param_0
)
{
.reg .b64 %rd<7>;


ld.param.u64 %rd1, [_Z24codegen_reset_force_halfP4Body_param_0];
shl.b64 %rd2, %rd1, 3;
mov.u64 %rd3, __Bodydata_buffer;
cvta.global.u64 %rd4, %rd3;
add.s64 %rd5, %rd4, %rd2;
mov.u64 %rd6, 0;
st.u64 [%rd5+320016], %rd6;
ret;
}


.visible .entry _Z18codegen_reset_massP4Body(
.param .u64 _Z18codegen_reset_massP4Body_param_0
)
{
.reg .b64 %rd<7>;


ld.param.u64 %rd1, [_Z18codegen_reset_massP4Body_param_0];
shl.b64 %rd2, %rd1, 3;
mov.u64 %rd3, __Bodydata_buffer;
cvta.global.u64 %rd4, %rd3;
add.s64 %rd5, %rd4, %rd2;
mov.u64 %rd6, 0;
st.u64 [%rd5+16], %rd6;
ret;
}


.visible .entry _Z22codengen_simple_updateP4Bodyd(
.param .u64 _Z22codengen_simple_updateP4Bodyd_param_0,
.param .f64 _Z22codengen_simple_updateP4Bodyd_param_1
)
{
.reg .f64 %fd<19>;
.reg .b64 %rd<6>;


ld.param.u64 %rd1, [_Z22codengen_simple_updateP4Bodyd_param_0];
ld.param.f64 %fd1, [_Z22codengen_simple_updateP4Bodyd_param_1];
shl.b64 %rd2, %rd1, 3;
mov.u64 %rd3, __Bodydata_buffer;
cvta.global.u64 %rd4, %rd3;
add.s64 %rd5, %rd4, %rd2;
ld.f64 %fd2, [%rd5+320016];
mul.f64 %fd3, %fd2, %fd1;
ld.f64 %fd4, [%rd5+16];
div.rn.f64 %fd5, %fd3, %fd4;
ld.f64 %fd6, [%rd5+192016];
ld.f64 %fd7, [%rd5+384016];
ld.f64 %fd8, [%rd5+256016];
add.f64 %fd9, %fd5, %fd6;
ld.f64 %fd10, [%rd5+64016];
ld.f64 %fd11, [%rd5+128016];
st.f64 [%rd5+192016], %fd9;
mul.f64 %fd12, %fd7, %fd1;
ld.f64 %fd13, [%rd5+16];
div.rn.f64 %fd14, %fd12, %fd13;
add.f64 %fd15, %fd14, %fd8;
st.f64 [%rd5+256016], %fd15;
fma.rn.f64 %fd16, %fd9, %fd1, %fd10;
st.f64 [%rd5+64016], %fd16;
ld.f64 %fd17, [%rd5+256016];
fma.rn.f64 %fd18, %fd17, %fd1, %fd11;
st.f64 [%rd5+128016], %fd18;
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
@%p1 bra BB8_2;

cvta.to.global.u64 %rd2, %rd1;
mov.u64 %rd3, 0;
st.global.u64 [%rd2], %rd3;
st.global.u64 [%rd2+8], %rd3;

BB8_2:
ret;
}


.visible .entry _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE_clEvEUlPT_DpT0_E_JP4BodyEEEvS4_mS7_(
.param .align 8 .b8 _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE_clEvEUlPT_DpT0_E_JP4BodyEEEvS4_mS7__param_0[8],
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE_clEvEUlPT_DpT0_E_JP4BodyEEEvS4_mS7__param_1,
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE_clEvEUlPT_DpT0_E_JP4BodyEEEvS4_mS7__param_2
)
{
.reg .pred %p<2>;
.reg .b32 %r<5>;
.reg .b64 %rd<10>;


ld.param.u64 %rd3, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE_clEvEUlPT_DpT0_E_JP4BodyEEEvS4_mS7__param_0];
ld.param.u64 %rd2, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE_clEvEUlPT_DpT0_E_JP4BodyEEEvS4_mS7__param_2];
mov.u32 %r1, %ntid.x;
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %tid.x;
mad.lo.s32 %r4, %r1, %r2, %r3;
cvt.s64.s32	%rd1, %r4;
setp.ge.u64	%p1, %rd1, %rd3;
@%p1 bra BB9_2;

add.s64 %rd4, %rd1, %rd2;
shl.b64 %rd5, %rd4, 3;
mov.u64 %rd6, __Bodydata_buffer;
cvta.global.u64 %rd7, %rd6;
add.s64 %rd8, %rd7, %rd5;
mov.u64 %rd9, 0;
st.u64 [%rd8+320016], %rd9;
st.u64 [%rd8+384016], %rd9;

BB9_2:
ret;
}


.visible .entry _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE0_clEvEUlPT_DpT0_E_JP4BodyEEEvS4_mS7_(
.param .align 8 .b8 _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE0_clEvEUlPT_DpT0_E_JP4BodyEEEvS4_mS7__param_0[8],
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE0_clEvEUlPT_DpT0_E_JP4BodyEEEvS4_mS7__param_1,
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE0_clEvEUlPT_DpT0_E_JP4BodyEEEvS4_mS7__param_2
)
{
.reg .pred %p<7>;
.reg .b32 %r<21>;
.reg .f64 %fd<93>;
.reg .b64 %rd<72>;


ld.param.u64 %rd19, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE0_clEvEUlPT_DpT0_E_JP4BodyEEEvS4_mS7__param_0];
ld.param.u64 %rd18, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE0_clEvEUlPT_DpT0_E_JP4BodyEEEvS4_mS7__param_2];
mov.u32 %r1, %ntid.x;
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %tid.x;
mad.lo.s32 %r4, %r1, %r2, %r3;
cvt.s64.s32	%rd1, %r4;
setp.ge.u64	%p1, %rd1, %rd19;
@%p1 bra BB10_11;

add.s64 %rd2, %rd1, %rd18;
shl.b64 %rd26, %rd2, 3;
mov.u64 %rd27, __Bodydata_buffer;
cvta.global.u64 %rd28, %rd27;
add.s64 %rd70, %rd28, 16;
add.s64 %rd29, %rd70, %rd26;
add.s64 %rd3, %rd29, 64000;
add.s64 %rd4, %rd29, 128000;
add.s64 %rd5, %rd29, 384000;
mov.u64 %rd71, 8000;
mov.u64 %rd69, 4;
mov.u64 %rd68, 3;
mov.u64 %rd67, 2;
mov.u64 %rd66, 1;

BB10_2:
setp.eq.s64	%p2, %rd2, %rd66;
@%p2 bra BB10_4;

add.s64 %rd35, %rd28, 16;
add.s64 %rd36, %rd26, %rd35;
shl.b64 %rd37, %rd66, 3;
ld.f64 %fd1, [%rd3];
add.s64 %rd38, %rd35, %rd37;
ld.f64 %fd2, [%rd38+64000];
sub.f64 %fd3, %fd2, %fd1;
ld.f64 %fd4, [%rd4];
ld.f64 %fd5, [%rd38+128000];
sub.f64 %fd6, %fd5, %fd4;
mul.f64 %fd7, %fd6, %fd6;
fma.rn.f64 %fd8, %fd3, %fd3, %fd7;
sqrt.rn.f64 %fd9, %fd8;
ld.f64 %fd10, [%rd36];
mul.f64 %fd11, %fd10, 0d3DD257B4D002790D;
ld.f64 %fd12, [%rd70+8];
mul.f64 %fd13, %fd11, %fd12;
fma.rn.f64 %fd14, %fd9, %fd9, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd15, %fd13, %fd14;
mul.f64 %fd16, %fd3, %fd15;
div.rn.f64 %fd17, %fd16, %fd9;
ld.f64 %fd18, [%rd36+320000];
add.f64 %fd19, %fd17, %fd18;
st.f64 [%rd36+320000], %fd19;
mul.f64 %fd20, %fd6, %fd15;
div.rn.f64 %fd21, %fd20, %fd9;
ld.f64 %fd22, [%rd5];
add.f64 %fd23, %fd21, %fd22;
st.f64 [%rd5], %fd23;

BB10_4:
setp.eq.s64	%p3, %rd2, %rd67;
@%p3 bra BB10_6;

add.s64 %rd44, %rd28, 16;
add.s64 %rd45, %rd26, %rd44;
shl.b64 %rd46, %rd67, 3;
ld.f64 %fd24, [%rd3];
add.s64 %rd47, %rd44, %rd46;
ld.f64 %fd25, [%rd47+64000];
sub.f64 %fd26, %fd25, %fd24;
ld.f64 %fd27, [%rd4];
ld.f64 %fd28, [%rd47+128000];
sub.f64 %fd29, %fd28, %fd27;
mul.f64 %fd30, %fd29, %fd29;
fma.rn.f64 %fd31, %fd26, %fd26, %fd30;
sqrt.rn.f64 %fd32, %fd31;
ld.f64 %fd33, [%rd45];
mul.f64 %fd34, %fd33, 0d3DD257B4D002790D;
ld.f64 %fd35, [%rd70+16];
mul.f64 %fd36, %fd34, %fd35;
fma.rn.f64 %fd37, %fd32, %fd32, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd38, %fd36, %fd37;
mul.f64 %fd39, %fd26, %fd38;
div.rn.f64 %fd40, %fd39, %fd32;
ld.f64 %fd41, [%rd45+320000];
add.f64 %fd42, %fd40, %fd41;
st.f64 [%rd45+320000], %fd42;
mul.f64 %fd43, %fd29, %fd38;
div.rn.f64 %fd44, %fd43, %fd32;
ld.f64 %fd45, [%rd5];
add.f64 %fd46, %fd44, %fd45;
st.f64 [%rd5], %fd46;

BB10_6:
setp.eq.s64	%p4, %rd2, %rd68;
@%p4 bra BB10_8;

add.s64 %rd53, %rd28, 16;
add.s64 %rd54, %rd26, %rd53;
shl.b64 %rd55, %rd68, 3;
ld.f64 %fd47, [%rd3];
add.s64 %rd56, %rd53, %rd55;
ld.f64 %fd48, [%rd56+64000];
sub.f64 %fd49, %fd48, %fd47;
ld.f64 %fd50, [%rd4];
ld.f64 %fd51, [%rd56+128000];
sub.f64 %fd52, %fd51, %fd50;
mul.f64 %fd53, %fd52, %fd52;
fma.rn.f64 %fd54, %fd49, %fd49, %fd53;
sqrt.rn.f64 %fd55, %fd54;
ld.f64 %fd56, [%rd54];
mul.f64 %fd57, %fd56, 0d3DD257B4D002790D;
ld.f64 %fd58, [%rd70+24];
mul.f64 %fd59, %fd57, %fd58;
fma.rn.f64 %fd60, %fd55, %fd55, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd61, %fd59, %fd60;
mul.f64 %fd62, %fd49, %fd61;
div.rn.f64 %fd63, %fd62, %fd55;
ld.f64 %fd64, [%rd54+320000];
add.f64 %fd65, %fd63, %fd64;
st.f64 [%rd54+320000], %fd65;
mul.f64 %fd66, %fd52, %fd61;
div.rn.f64 %fd67, %fd66, %fd55;
ld.f64 %fd68, [%rd5];
add.f64 %fd69, %fd67, %fd68;
st.f64 [%rd5], %fd69;

BB10_8:
setp.eq.s64	%p5, %rd2, %rd69;
@%p5 bra BB10_10;

add.s64 %rd62, %rd28, 16;
add.s64 %rd63, %rd26, %rd62;
shl.b64 %rd64, %rd69, 3;
ld.f64 %fd70, [%rd3];
add.s64 %rd65, %rd62, %rd64;
ld.f64 %fd71, [%rd65+64000];
sub.f64 %fd72, %fd71, %fd70;
ld.f64 %fd73, [%rd4];
ld.f64 %fd74, [%rd65+128000];
sub.f64 %fd75, %fd74, %fd73;
mul.f64 %fd76, %fd75, %fd75;
fma.rn.f64 %fd77, %fd72, %fd72, %fd76;
sqrt.rn.f64 %fd78, %fd77;
ld.f64 %fd79, [%rd63];
mul.f64 %fd80, %fd79, 0d3DD257B4D002790D;
ld.f64 %fd81, [%rd70+32];
mul.f64 %fd82, %fd80, %fd81;
fma.rn.f64 %fd83, %fd78, %fd78, 0d3F1A36E2EB1C432D;
div.rn.f64 %fd84, %fd82, %fd83;
mul.f64 %fd85, %fd72, %fd84;
div.rn.f64 %fd86, %fd85, %fd78;
ld.f64 %fd87, [%rd63+320000];
add.f64 %fd88, %fd86, %fd87;
st.f64 [%rd63+320000], %fd88;
mul.f64 %fd89, %fd75, %fd84;
div.rn.f64 %fd90, %fd89, %fd78;
ld.f64 %fd91, [%rd5];
add.f64 %fd92, %fd90, %fd91;
st.f64 [%rd5], %fd92;

BB10_10:
add.s64 %rd71, %rd71, -4;
add.s64 %rd70, %rd70, 32;
add.s64 %rd69, %rd69, 4;
add.s64 %rd68, %rd68, 4;
add.s64 %rd67, %rd67, 4;
add.s64 %rd66, %rd66, 4;
setp.ne.s64	%p6, %rd71, 0;
@%p6 bra BB10_2;

BB10_11:
ret;
}


.visible .entry _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE1_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7_(
.param .align 8 .b8 _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE1_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7__param_0[8],
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE1_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7__param_1,
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE1_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7__param_2,
.param .f64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE1_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7__param_3
)
{
.reg .pred %p<8>;
.reg .b32 %r<5>;
.reg .f64 %fd<27>;
.reg .b64 %rd<12>;


ld.param.u64 %rd6, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE1_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7__param_0];
ld.param.u64 %rd5, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE1_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7__param_2];
ld.param.f64 %fd4, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZZ14run_simulationvENKUlvE1_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7__param_3];
mov.u32 %r1, %ntid.x;
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %tid.x;
mad.lo.s32 %r4, %r1, %r2, %r3;
cvt.s64.s32	%rd1, %r4;
setp.ge.u64	%p1, %rd1, %rd6;
@%p1 bra BB11_5;

add.s64 %rd7, %rd1, %rd5;
shl.b64 %rd8, %rd7, 3;
mov.u64 %rd9, __Bodydata_buffer;
cvta.global.u64 %rd10, %rd9;
add.s64 %rd11, %rd10, %rd8;
ld.f64 %fd5, [%rd11+320016];
mul.f64 %fd6, %fd5, %fd4;
ld.f64 %fd7, [%rd11+16];
div.rn.f64 %fd8, %fd6, %fd7;
add.s64 %rd2, %rd11, 192016;
ld.f64 %fd9, [%rd11+192016];
ld.f64 %fd10, [%rd11+384016];
ld.f64 %fd11, [%rd11+256016];
add.f64 %fd12, %fd8, %fd9;
ld.f64 %fd13, [%rd11+64016];
ld.f64 %fd14, [%rd11+128016];
st.f64 [%rd11+192016], %fd12;
mul.f64 %fd15, %fd10, %fd4;
ld.f64 %fd16, [%rd11+16];
div.rn.f64 %fd17, %fd15, %fd16;
add.s64 %rd3, %rd11, 256016;
add.f64 %fd18, %fd17, %fd11;
st.f64 [%rd11+256016], %fd18;
fma.rn.f64 %fd19, %fd12, %fd4, %fd13;
st.f64 [%rd11+64016], %fd19;
ld.f64 %fd20, [%rd11+256016];
add.s64 %rd4, %rd11, 128016;
fma.rn.f64 %fd26, %fd20, %fd4, %fd14;
ld.f64 %fd21, [%rd11+64016];
st.f64 [%rd11+128016], %fd26;
setp.lt.f64	%p2, %fd21, 0dBFF0000000000000;
setp.gt.f64	%p3, %fd21, 0d3FF0000000000000;
or.pred %p4, %p2, %p3;
@!%p4 bra BB11_3;
bra.uni BB11_2;

BB11_2:
ld.f64 %fd22, [%rd2];
neg.f64 %fd23, %fd22;
st.f64 [%rd2], %fd23;
ld.f64 %fd26, [%rd4];

BB11_3:
setp.gt.f64	%p5, %fd26, 0d3FF0000000000000;
setp.lt.f64	%p6, %fd26, 0dBFF0000000000000;
or.pred %p7, %p6, %p5;
@!%p7 bra BB11_5;
bra.uni BB11_4;

BB11_4:
ld.f64 %fd24, [%rd3];
neg.f64 %fd25, %fd24;
st.f64 [%rd3], %fd25;

BB11_5:
ret;
}


.visible .entry _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ10run_simplevENKUlvE_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7_(
.param .align 8 .b8 _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ10run_simplevENKUlvE_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7__param_0[8],
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ10run_simplevENKUlvE_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7__param_1,
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ10run_simplevENKUlvE_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7__param_2,
.param .f64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZZ10run_simplevENKUlvE_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7__param_3
)
{
.reg .pred %p<2>;
.reg .b32 %r<5>;
.reg .f64 %fd<19>;
.reg .b64 %rd<9>;


ld.param.u64 %rd3, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZZ10run_simplevENKUlvE_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7__param_0];
ld.param.u64 %rd2, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZZ10run_simplevENKUlvE_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7__param_2];
ld.param.f64 %fd1, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZZ10run_simplevENKUlvE_clEvEUlPT_DpT0_E_JP4BodydEEEvS4_mS7__param_3];
mov.u32 %r1, %ntid.x;
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %tid.x;
mad.lo.s32 %r4, %r1, %r2, %r3;
cvt.s64.s32	%rd1, %r4;
setp.ge.u64	%p1, %rd1, %rd3;
@%p1 bra BB12_2;

add.s64 %rd4, %rd1, %rd2;
shl.b64 %rd5, %rd4, 3;
mov.u64 %rd6, __Bodydata_buffer;
cvta.global.u64 %rd7, %rd6;
add.s64 %rd8, %rd7, %rd5;
ld.f64 %fd2, [%rd8+320016];
mul.f64 %fd3, %fd2, %fd1;
ld.f64 %fd4, [%rd8+16];
div.rn.f64 %fd5, %fd3, %fd4;
ld.f64 %fd6, [%rd8+192016];
ld.f64 %fd7, [%rd8+384016];
ld.f64 %fd8, [%rd8+256016];
add.f64 %fd9, %fd5, %fd6;
ld.f64 %fd10, [%rd8+64016];
ld.f64 %fd11, [%rd8+128016];
st.f64 [%rd8+192016], %fd9;
mul.f64 %fd12, %fd7, %fd1;
ld.f64 %fd13, [%rd8+16];
div.rn.f64 %fd14, %fd12, %fd13;
add.f64 %fd15, %fd14, %fd8;
st.f64 [%rd8+256016], %fd15;
fma.rn.f64 %fd16, %fd9, %fd1, %fd10;
st.f64 [%rd8+64016], %fd16;
ld.f64 %fd17, [%rd8+256016];
fma.rn.f64 %fd18, %fd17, %fd1, %fd11;
st.f64 [%rd8+128016], %fd18;

BB12_2:
ret;
}


