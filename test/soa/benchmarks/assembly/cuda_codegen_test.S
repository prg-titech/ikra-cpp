
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
  GLOBAL:8452
 Function _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I9TestClassLm0EEEJEEEvPT_DpT0_:
  REG:4 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra3soa28storage_increase_size_kernelEPNS0_19StorageStrategyBaseEm:
  REG:10 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM9TestClassFviEXadL_ZNS4_10add_fieldsEiEEE4callEiEUliE_JiEEEvT_DpT0_:
  REG:8 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda16construct_kernelI9TestClassJiiEEEvmDpT0_:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda27increment_class_size_kernelI9TestClassEEvm:
  REG:14 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z20explicit_read_field0m:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z21explicit_write_field0m:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z11read_field1P9TestClass:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z12write_field1P9TestClass:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z11read_field0P9TestClass:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z12write_field0P9TestClass:
  REG:6 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0

	code for sm_30
		Function : _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I9TestClassLm0EEEJEEEvPT_DpT0_
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
		..........................................................................................................


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


		Function : _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM9TestClassFviEXadL_ZNS4_10add_fieldsEiEEE4callEiEUliE_JiEEEvT_DpT0_
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
        /*0078*/                   MOV32I R0, 0x2;                         /* 0x1800000008001de2 */
                                                                           /* 0x2202b20242323047 */
        /*0088*/                   IADD32I RZ.CC, R0, -0x20;               /* 0x0fffffff800fdc02 */
        /*0090*/                   SHL R5, R5, 0x2;                        /* 0x6000c00008515c03 */
        /*0098*/                   SHL R0, R4, -0x1e;                      /* 0x6000ffff88401c03 */
        /*00a0*/                   SHR.U32 R2, R4, 0x1e;                   /* 0x5800c00078409c03 */
        /*00a8*/                   CSETP.SFT.AND P0, PT, CC, PT;           /* 0x040e00000151dc04 */
        /*00b0*/                   IMAD.U32.U32 R7.CC, R4, 0x4, R7;        /* 0x200fc0001041dc03 */
        /*00b8*/               @P0 IADD R0, R5, R2;                        /* 0x4800000008500003 */
                                                                           /* 0x22828283f2328277 */
        /*00c8*/                   MOV R2, R7;                             /* 0x280000001c009de4 */
        /*00d0*/                   IADD.X R3, R0, R6;                      /* 0x480000001800dc43 */
        /*00d8*/                   LD.E R0, [R2+0x14];                     /* 0x8400000050201c85 */
        /*00e0*/                   LD.E R5, [R2+0xfb4];                    /* 0x8400003ed0215c85 */
        /*00e8*/                   IADD R0, R0, c[0x0][0x144];             /* 0x4800400510001c03 */
        /*00f0*/                   IADD R0, R0, R5;                        /* 0x4800000014001c03 */
        /*00f8*/                   IADD R0, R4, R0;                        /* 0x4800000000401c03 */
                                                                           /* 0x2000000002f04287 */
        /*0108*/                   IADD32I R4, R0, 0x7777;                 /* 0x080001dddc011c02 */
        /*0110*/                   ST.E [R2+0x14], R4;                     /* 0x9400000050211c85 */
        /*0118*/                   EXIT;                                   /* 0x8000000000001de7 */
        /*0120*/                   BRA 0x120;                              /* 0x4003ffffe0001de7 */
        /*0128*/                   NOP;                                    /* 0x4000000000001de4 */
        /*0130*/                   NOP;                                    /* 0x4000000000001de4 */
        /*0138*/                   NOP;                                    /* 0x4000000000001de4 */
		....................................................................................................................................................


		Function : _ZN4ikra8executor4cuda16construct_kernelI9TestClassJiiEEEvmDpT0_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                         /* 0x2202c28042823307 */
        /*0008*/                   MOV R1, c[0x0][0x44];                 /* 0x2800400110005de4 */
        /*0010*/                   S2R R0, SR_TID.X;                     /* 0x2c00000084001c04 */
        /*0018*/                   S2R R3, SR_CTAID.X;                   /* 0x2c0000009400dc04 */
        /*0020*/                   IMAD R0, R3, c[0x0][0x28], R0;        /* 0x20004000a0301ca3 */
        /*0028*/                   MOV32I R3, 0x2;                       /* 0x180000000800dde2 */
        /*0030*/                   IADD R4.CC, R0, c[0x0][0x140];        /* 0x4801400500011c03 */
        /*0038*/                   IMAD.HI.X R0, R0, R3, c[0x0][0x144];  /* 0x2086800510001ce3 */
                                                                         /* 0x228042f2c2c042c7 */
        /*0048*/                   IADD32I R4.CC, R4, 0x1;               /* 0x0c00000004411c02 */
        /*0050*/                   IADD.X R0, RZ, R0;                    /* 0x4800000003f01c43 */
        /*0058*/                   IADD RZ.CC, -RZ, R4;                  /* 0x4801000013ffde03 */
        /*0060*/                   ISETP.EQ.X.AND P0, PT, R0, RZ, PT;    /* 0x190e0000fc01dc63 */
        /*0068*/               @P0 EXIT;                                 /* 0x80000000000001e7 */
        /*0070*/                   MOV32I R2, 0x2;                       /* 0x1800000008009de2 */
        /*0078*/                   SHL R3, R4, -0x1e;                    /* 0x6000ffff8840dc03 */
                                                                         /* 0x22b0420272304207 */
        /*0088*/                   IADD32I RZ.CC, R2, -0x20;             /* 0x0fffffff802fdc02 */
        /*0090*/                   SHL R2, R0, 0x2;                      /* 0x6000c00008009c03 */
        /*0098*/                   MOV32I R0, 0x0;                       /* 0x1800000000001de2 */
        /*00a0*/                   SHR.U32 R5, R4, 0x1e;                 /* 0x5800c00078415c03 */
        /*00a8*/                   CSETP.SFT.AND P0, PT, CC, PT;         /* 0x040e00000151dc04 */
        /*00b0*/                   IMAD.U32.U32 R4.CC, R4, 0x4, R0;      /* 0x2001c00010411c03 */
        /*00b8*/                   MOV32I R0, 0x0;                       /* 0x1800000000001de2 */
                                                                         /* 0x2202e28262020047 */
        /*00c8*/               @P0 IADD R3, R2, R5;                      /* 0x480000001420c003 */
        /*00d0*/                   MOV R5, c[0x0][0x148];                /* 0x2800400520015de4 */
        /*00d8*/                   MOV R2, R4;                           /* 0x2800000010009de4 */
        /*00e0*/                   MOV R4, c[0x0][0x14c];                /* 0x2800400530011de4 */
        /*00e8*/                   IADD.X R3, R0, R3;                    /* 0x480000000c00dc43 */
        /*00f0*/                   ST.E [R2+0x10], R5;                   /* 0x9400000040215c85 */
        /*00f8*/                   ST.E [R2+0xfb0], R4;                  /* 0x9400003ec0211c85 */
                                                                         /* 0x20000000000002f7 */
        /*0108*/                   EXIT;                                 /* 0x8000000000001de7 */
        /*0110*/                   BRA 0x110;                            /* 0x4003ffffe0001de7 */
        /*0118*/                   NOP;                                  /* 0x4000000000001de4 */
        /*0120*/                   NOP;                                  /* 0x4000000000001de4 */
        /*0128*/                   NOP;                                  /* 0x4000000000001de4 */
        /*0130*/                   NOP;                                  /* 0x4000000000001de4 */
        /*0138*/                   NOP;                                  /* 0x4000000000001de4 */
		...........................................................................


		Function : _ZN4ikra8executor4cuda27increment_class_size_kernelI9TestClassEEvm
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                    /* 0x2202020042820307 */
        /*0008*/                   MOV R1, c[0x0][0x44];            /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R2, 0x0;                  /* 0x1800000000009de2 */
        /*0018*/                   MOV32I R3, 0x0;                  /* 0x180000000000dde2 */
        /*0020*/                   LD.E.64 R4, [R2];                /* 0x8400000000211ca5 */
        /*0028*/                   MOV32I R6, 0x0;                  /* 0x1800000000019de2 */
        /*0030*/                   MOV32I R7, 0x0;                  /* 0x180000000001dde2 */
        /*0038*/                   MOV32I R8, 0x0;                  /* 0x1800000000021de2 */
                                                                    /* 0x22e2e042c042c3f7 */
        /*0048*/                   MOV32I R9, 0x0;                  /* 0x1800000000025de2 */
        /*0050*/                   IADD32I R10.CC, R4, 0x1;         /* 0x0c00000004429c02 */
        /*0058*/                   IADD.X R11, RZ, R5;              /* 0x4800000017f2dc43 */
        /*0060*/                   IADD R12.CC, R4, c[0x0][0x140];  /* 0x4801400500431c03 */
        /*0068*/                   IADD.X R13, R5, c[0x0][0x144];   /* 0x4800400510535c43 */
        /*0070*/                   ST.E.64 [R6], R10;               /* 0x9400000000629ca5 */
        /*0078*/                   ST.E.64 [R8], R4;                /* 0x9400000000811ca5 */
                                                                    /* 0x200000000002f047 */
        /*0088*/                   ST.E.64 [R2], R12;               /* 0x9400000000231ca5 */
        /*0090*/                   EXIT;                            /* 0x8000000000001de7 */
        /*0098*/                   BRA 0x98;                        /* 0x4003ffffe0001de7 */
        /*00a0*/                   NOP;                             /* 0x4000000000001de4 */
        /*00a8*/                   NOP;                             /* 0x4000000000001de4 */
        /*00b0*/                   NOP;                             /* 0x4000000000001de4 */
        /*00b8*/                   NOP;                             /* 0x4000000000001de4 */
		.............................................................................


		Function : _Z20explicit_read_field0m
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                     /* 0x2240423042804307 */
        /*0008*/                   MOV R1, c[0x0][0x44];             /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R0, 0x2;                   /* 0x1800000008001de2 */
        /*0018*/                   MOV R2, c[0x0][0x144];            /* 0x2800400510009de4 */
        /*0020*/                   MOV R5, c[0x0][0x140];            /* 0x2800400500015de4 */
        /*0028*/                   IADD32I RZ.CC, R0, -0x20;         /* 0x0fffffff800fdc02 */
        /*0030*/                   MOV32I R4, 0x0;                   /* 0x1800000000011de2 */
        /*0038*/                   SHL R2, R2, 0x2;                  /* 0x6000c00008209c03 */
                                                                     /* 0x2282827042023237 */
        /*0048*/                   SHL R0, R5, -0x1e;                /* 0x6000ffff88501c03 */
        /*0050*/                   CSETP.SFT.AND P0, PT, CC, PT;     /* 0x040e00000151dc04 */
        /*0058*/                   SHR.U32 R3, R5, 0x1e;             /* 0x5800c0007850dc03 */
        /*0060*/                   IMAD.U32.U32 R4.CC, R5, 0x4, R4;  /* 0x2009c00010511c03 */
        /*0068*/                   MOV32I R5, 0x0;                   /* 0x1800000000015de2 */
        /*0070*/               @P0 IADD R0, R2, R3;                  /* 0x480000000c200003 */
        /*0078*/                   IADD.X R5, R5, R0;                /* 0x4800000000515c43 */
                                                                     /* 0x200002f043f20047 */
        /*0088*/                   LD.E R4, [R4];                    /* 0x8400000000411c85 */
        /*0090*/                   MOV32I R2, 0x0;                   /* 0x1800000000009de2 */
        /*0098*/                   MOV32I R3, 0x0;                   /* 0x180000000000dde2 */
        /*00a0*/                   ST.E [R2], R4;                    /* 0x9400000000211c85 */
        /*00a8*/                   EXIT;                             /* 0x8000000000001de7 */
        /*00b0*/                   BRA 0xb0;                         /* 0x4003ffffe0001de7 */
        /*00b8*/                   NOP;                              /* 0x4000000000001de4 */
		....................................


		Function : _Z21explicit_write_field0m
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                     /* 0x2240423042804307 */
        /*0008*/                   MOV R1, c[0x0][0x44];             /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R0, 0x2;                   /* 0x1800000008001de2 */
        /*0018*/                   MOV R4, c[0x0][0x144];            /* 0x2800400510011de4 */
        /*0020*/                   MOV R3, c[0x0][0x140];            /* 0x280040050000dde4 */
        /*0028*/                   IADD32I RZ.CC, R0, -0x20;         /* 0x0fffffff800fdc02 */
        /*0030*/                   MOV32I R2, 0x0;                   /* 0x1800000000009de2 */
        /*0038*/                   SHL R4, R4, 0x2;                  /* 0x6000c00008411c03 */
                                                                     /* 0x2280427042023237 */
        /*0048*/                   SHL R0, R3, -0x1e;                /* 0x6000ffff88301c03 */
        /*0050*/                   CSETP.SFT.AND P0, PT, CC, PT;     /* 0x040e00000151dc04 */
        /*0058*/                   SHR.U32 R5, R3, 0x1e;             /* 0x5800c00078315c03 */
        /*0060*/                   IMAD.U32.U32 R2.CC, R3, 0x4, R2;  /* 0x2005c00010309c03 */
        /*0068*/                   MOV32I R3, 0x0;                   /* 0x180000000000dde2 */
        /*0070*/               @P0 IADD R0, R4, R5;                  /* 0x4800000014400003 */
        /*0078*/                   MOV32I R4, 0x7777;                /* 0x180001dddc011de2 */
                                                                     /* 0x2000000002f04287 */
        /*0088*/                   IADD.X R3, R3, R0;                /* 0x480000000030dc43 */
        /*0090*/                   ST.E [R2], R4;                    /* 0x9400000000211c85 */
        /*0098*/                   EXIT;                             /* 0x8000000000001de7 */
        /*00a0*/                   BRA 0xa0;                         /* 0x4003ffffe0001de7 */
        /*00a8*/                   NOP;                              /* 0x4000000000001de4 */
        /*00b0*/                   NOP;                              /* 0x4000000000001de4 */
        /*00b8*/                   NOP;                              /* 0x4000000000001de4 */
		.....................................


		Function : _Z11read_field1P9TestClass
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                     /* 0x2240423042804307 */
        /*0008*/                   MOV R1, c[0x0][0x44];             /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R0, 0x2;                   /* 0x1800000008001de2 */
        /*0018*/                   MOV R2, c[0x0][0x144];            /* 0x2800400510009de4 */
        /*0020*/                   MOV R5, c[0x0][0x140];            /* 0x2800400500015de4 */
        /*0028*/                   IADD32I RZ.CC, R0, -0x20;         /* 0x0fffffff800fdc02 */
        /*0030*/                   MOV32I R4, 0x0;                   /* 0x1800000000011de2 */
        /*0038*/                   SHL R2, R2, 0x2;                  /* 0x6000c00008209c03 */
                                                                     /* 0x2282827042023237 */
        /*0048*/                   SHL R0, R5, -0x1e;                /* 0x6000ffff88501c03 */
        /*0050*/                   CSETP.SFT.AND P0, PT, CC, PT;     /* 0x040e00000151dc04 */
        /*0058*/                   SHR.U32 R3, R5, 0x1e;             /* 0x5800c0007850dc03 */
        /*0060*/                   IMAD.U32.U32 R4.CC, R5, 0x4, R4;  /* 0x2009c00010511c03 */
        /*0068*/                   MOV32I R5, 0x0;                   /* 0x1800000000015de2 */
        /*0070*/               @P0 IADD R0, R2, R3;                  /* 0x480000000c200003 */
        /*0078*/                   IADD.X R5, R5, R0;                /* 0x4800000000515c43 */
                                                                     /* 0x200002f043f20047 */
        /*0088*/                   LD.E R4, [R4+0xfb0];              /* 0x8400003ec0411c85 */
        /*0090*/                   MOV32I R2, 0x0;                   /* 0x1800000000009de2 */
        /*0098*/                   MOV32I R3, 0x0;                   /* 0x180000000000dde2 */
        /*00a0*/                   ST.E [R2], R4;                    /* 0x9400000000211c85 */
        /*00a8*/                   EXIT;                             /* 0x8000000000001de7 */
        /*00b0*/                   BRA 0xb0;                         /* 0x4003ffffe0001de7 */
        /*00b8*/                   NOP;                              /* 0x4000000000001de4 */
		.....................................


		Function : _Z12write_field1P9TestClass
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                     /* 0x2240423042804307 */
        /*0008*/                   MOV R1, c[0x0][0x44];             /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R0, 0x2;                   /* 0x1800000008001de2 */
        /*0018*/                   MOV R4, c[0x0][0x144];            /* 0x2800400510011de4 */
        /*0020*/                   MOV R3, c[0x0][0x140];            /* 0x280040050000dde4 */
        /*0028*/                   IADD32I RZ.CC, R0, -0x20;         /* 0x0fffffff800fdc02 */
        /*0030*/                   MOV32I R2, 0x0;                   /* 0x1800000000009de2 */
        /*0038*/                   SHL R4, R4, 0x2;                  /* 0x6000c00008411c03 */
                                                                     /* 0x2280427042023237 */
        /*0048*/                   SHL R0, R3, -0x1e;                /* 0x6000ffff88301c03 */
        /*0050*/                   CSETP.SFT.AND P0, PT, CC, PT;     /* 0x040e00000151dc04 */
        /*0058*/                   SHR.U32 R5, R3, 0x1e;             /* 0x5800c00078315c03 */
        /*0060*/                   IMAD.U32.U32 R2.CC, R3, 0x4, R2;  /* 0x2005c00010309c03 */
        /*0068*/                   MOV32I R3, 0x0;                   /* 0x180000000000dde2 */
        /*0070*/               @P0 IADD R0, R4, R5;                  /* 0x4800000014400003 */
        /*0078*/                   MOV32I R4, 0x7777;                /* 0x180001dddc011de2 */
                                                                     /* 0x2000000002f04287 */
        /*0088*/                   IADD.X R3, R3, R0;                /* 0x480000000030dc43 */
        /*0090*/                   ST.E [R2+0xfb0], R4;              /* 0x9400003ec0211c85 */
        /*0098*/                   EXIT;                             /* 0x8000000000001de7 */
        /*00a0*/                   BRA 0xa0;                         /* 0x4003ffffe0001de7 */
        /*00a8*/                   NOP;                              /* 0x4000000000001de4 */
        /*00b0*/                   NOP;                              /* 0x4000000000001de4 */
        /*00b8*/                   NOP;                              /* 0x4000000000001de4 */
		......................................


		Function : _Z11read_field0P9TestClass
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                     /* 0x2240423042804307 */
        /*0008*/                   MOV R1, c[0x0][0x44];             /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R0, 0x2;                   /* 0x1800000008001de2 */
        /*0018*/                   MOV R2, c[0x0][0x144];            /* 0x2800400510009de4 */
        /*0020*/                   MOV R5, c[0x0][0x140];            /* 0x2800400500015de4 */
        /*0028*/                   IADD32I RZ.CC, R0, -0x20;         /* 0x0fffffff800fdc02 */
        /*0030*/                   MOV32I R4, 0x0;                   /* 0x1800000000011de2 */
        /*0038*/                   SHL R2, R2, 0x2;                  /* 0x6000c00008209c03 */
                                                                     /* 0x2282827042023237 */
        /*0048*/                   SHL R0, R5, -0x1e;                /* 0x6000ffff88501c03 */
        /*0050*/                   CSETP.SFT.AND P0, PT, CC, PT;     /* 0x040e00000151dc04 */
        /*0058*/                   SHR.U32 R3, R5, 0x1e;             /* 0x5800c0007850dc03 */
        /*0060*/                   IMAD.U32.U32 R4.CC, R5, 0x4, R4;  /* 0x2009c00010511c03 */
        /*0068*/                   MOV32I R5, 0x0;                   /* 0x1800000000015de2 */
        /*0070*/               @P0 IADD R0, R2, R3;                  /* 0x480000000c200003 */
        /*0078*/                   IADD.X R5, R5, R0;                /* 0x4800000000515c43 */
                                                                     /* 0x200002f043f20047 */
        /*0088*/                   LD.E R4, [R4+0x10];               /* 0x8400000040411c85 */
        /*0090*/                   MOV32I R2, 0x0;                   /* 0x1800000000009de2 */
        /*0098*/                   MOV32I R3, 0x0;                   /* 0x180000000000dde2 */
        /*00a0*/                   ST.E [R2], R4;                    /* 0x9400000000211c85 */
        /*00a8*/                   EXIT;                             /* 0x8000000000001de7 */
        /*00b0*/                   BRA 0xb0;                         /* 0x4003ffffe0001de7 */
        /*00b8*/                   NOP;                              /* 0x4000000000001de4 */
		.....................................


		Function : _Z12write_field0P9TestClass
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                     /* 0x2240423042804307 */
        /*0008*/                   MOV R1, c[0x0][0x44];             /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R0, 0x2;                   /* 0x1800000008001de2 */
        /*0018*/                   MOV R4, c[0x0][0x144];            /* 0x2800400510011de4 */
        /*0020*/                   MOV R3, c[0x0][0x140];            /* 0x280040050000dde4 */
        /*0028*/                   IADD32I RZ.CC, R0, -0x20;         /* 0x0fffffff800fdc02 */
        /*0030*/                   MOV32I R2, 0x0;                   /* 0x1800000000009de2 */
        /*0038*/                   SHL R4, R4, 0x2;                  /* 0x6000c00008411c03 */
                                                                     /* 0x2280427042023237 */
        /*0048*/                   SHL R0, R3, -0x1e;                /* 0x6000ffff88301c03 */
        /*0050*/                   CSETP.SFT.AND P0, PT, CC, PT;     /* 0x040e00000151dc04 */
        /*0058*/                   SHR.U32 R5, R3, 0x1e;             /* 0x5800c00078315c03 */
        /*0060*/                   IMAD.U32.U32 R2.CC, R3, 0x4, R2;  /* 0x2005c00010309c03 */
        /*0068*/                   MOV32I R3, 0x0;                   /* 0x180000000000dde2 */
        /*0070*/               @P0 IADD R0, R4, R5;                  /* 0x4800000014400003 */
        /*0078*/                   MOV32I R4, 0x7777;                /* 0x180001dddc011de2 */
                                                                     /* 0x2000000002f04287 */
        /*0088*/                   IADD.X R3, R3, R0;                /* 0x480000000030dc43 */
        /*0090*/                   ST.E [R2+0x10], R4;               /* 0x9400000040211c85 */
        /*0098*/                   EXIT;                             /* 0x8000000000001de7 */
        /*00a0*/                   BRA 0xa0;                         /* 0x4003ffffe0001de7 */
        /*00a8*/                   NOP;                              /* 0x4000000000001de4 */
        /*00b0*/                   NOP;                              /* 0x4000000000001de4 */
        /*00b8*/                   NOP;                              /* 0x4000000000001de4 */
		......................................



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


.global .align 16 .b8 __TestClassdata_buffer[8024];
.global .align 4 .u32 int_result;
.global .align 16 .b8 explicit_field0[400];
.global .align 8 .u64 _ZN4ikra8executor4cuda14d_storage_sizeE;
.global .align 8 .u64 _ZN4ikra8executor4cuda19d_storage_data_headE;
.global .align 8 .u64 _ZN4ikra3soa23d_previous_storage_sizeE;

.visible .entry _Z12write_field0P9TestClass(
.param .u64 _Z12write_field0P9TestClass_param_0
)
{
.reg .b32 %r<2>;
.reg .b64 %rd<5>;


ld.param.u64 %rd1, [_Z12write_field0P9TestClass_param_0];
shl.b64 %rd2, %rd1, 2;
mov.u64 %rd3, __TestClassdata_buffer;
add.s64 %rd4, %rd3, %rd2;
mov.u32 %r1, 30583;
st.global.u32 [%rd4+16], %r1;
ret;
}


.visible .entry _Z11read_field0P9TestClass(
.param .u64 _Z11read_field0P9TestClass_param_0
)
{
.reg .b32 %r<2>;
.reg .b64 %rd<5>;


ld.param.u64 %rd1, [_Z11read_field0P9TestClass_param_0];
shl.b64 %rd2, %rd1, 2;
mov.u64 %rd3, __TestClassdata_buffer;
add.s64 %rd4, %rd3, %rd2;
ld.global.u32 %r1, [%rd4+16];
st.global.u32 [int_result], %r1;
ret;
}


.visible .entry _Z12write_field1P9TestClass(
.param .u64 _Z12write_field1P9TestClass_param_0
)
{
.reg .b32 %r<2>;
.reg .b64 %rd<5>;


ld.param.u64 %rd1, [_Z12write_field1P9TestClass_param_0];
shl.b64 %rd2, %rd1, 2;
mov.u64 %rd3, __TestClassdata_buffer;
add.s64 %rd4, %rd3, %rd2;
mov.u32 %r1, 30583;
st.global.u32 [%rd4+4016], %r1;
ret;
}


.visible .entry _Z11read_field1P9TestClass(
.param .u64 _Z11read_field1P9TestClass_param_0
)
{
.reg .b32 %r<2>;
.reg .b64 %rd<5>;


ld.param.u64 %rd1, [_Z11read_field1P9TestClass_param_0];
shl.b64 %rd2, %rd1, 2;
mov.u64 %rd3, __TestClassdata_buffer;
add.s64 %rd4, %rd3, %rd2;
ld.global.u32 %r1, [%rd4+4016];
st.global.u32 [int_result], %r1;
ret;
}


.visible .entry _Z21explicit_write_field0m(
.param .u64 _Z21explicit_write_field0m_param_0
)
{
.reg .b32 %r<2>;
.reg .b64 %rd<5>;


ld.param.u64 %rd1, [_Z21explicit_write_field0m_param_0];
shl.b64 %rd2, %rd1, 2;
mov.u64 %rd3, explicit_field0;
add.s64 %rd4, %rd3, %rd2;
mov.u32 %r1, 30583;
st.global.u32 [%rd4], %r1;
ret;
}


.visible .entry _Z20explicit_read_field0m(
.param .u64 _Z20explicit_read_field0m_param_0
)
{
.reg .b32 %r<2>;
.reg .b64 %rd<5>;


ld.param.u64 %rd1, [_Z20explicit_read_field0m_param_0];
shl.b64 %rd2, %rd1, 2;
mov.u64 %rd3, explicit_field0;
add.s64 %rd4, %rd3, %rd2;
ld.global.u32 %r1, [%rd4];
st.global.u32 [int_result], %r1;
ret;
}


.visible .entry _ZN4ikra8executor4cuda27increment_class_size_kernelI9TestClassEEvm(
.param .u64 _ZN4ikra8executor4cuda27increment_class_size_kernelI9TestClassEEvm_param_0
)
{
.reg .b64 %rd<5>;


ld.param.u64 %rd1, [_ZN4ikra8executor4cuda27increment_class_size_kernelI9TestClassEEvm_param_0];
ld.global.u64 %rd2, [__TestClassdata_buffer];
add.s64 %rd3, %rd2, 1;
st.global.u64 [_ZN4ikra8executor4cuda19d_storage_data_headE], %rd3;
st.global.u64 [_ZN4ikra8executor4cuda14d_storage_sizeE], %rd2;
add.s64 %rd4, %rd2, %rd1;
st.global.u64 [__TestClassdata_buffer], %rd4;
ret;
}


.visible .entry _ZN4ikra8executor4cuda16construct_kernelI9TestClassJiiEEEvmDpT0_(
.param .u64 _ZN4ikra8executor4cuda16construct_kernelI9TestClassJiiEEEvmDpT0__param_0,
.param .u32 _ZN4ikra8executor4cuda16construct_kernelI9TestClassJiiEEEvmDpT0__param_1,
.param .u32 _ZN4ikra8executor4cuda16construct_kernelI9TestClassJiiEEEvmDpT0__param_2
)
{
.reg .pred %p<2>;
.reg .b32 %r<7>;
.reg .b64 %rd<8>;


ld.param.u64 %rd2, [_ZN4ikra8executor4cuda16construct_kernelI9TestClassJiiEEEvmDpT0__param_0];
ld.param.u32 %r1, [_ZN4ikra8executor4cuda16construct_kernelI9TestClassJiiEEEvmDpT0__param_1];
ld.param.u32 %r2, [_ZN4ikra8executor4cuda16construct_kernelI9TestClassJiiEEEvmDpT0__param_2];
mov.u32 %r3, %tid.x;
mov.u32 %r4, %ntid.x;
mov.u32 %r5, %ctaid.x;
mad.lo.s32 %r6, %r4, %r5, %r3;
cvt.s64.s32	%rd3, %r6;
add.s64 %rd4, %rd2, %rd3;
add.s64 %rd1, %rd4, 1;
setp.eq.s64	%p1, %rd1, 0;
@%p1 bra BB7_2;

shl.b64 %rd5, %rd1, 2;
mov.u64 %rd6, __TestClassdata_buffer;
add.s64 %rd7, %rd6, %rd5;
st.global.u32 [%rd7+16], %r1;
st.global.u32 [%rd7+4016], %r2;

BB7_2:
ret;
}


.visible .entry _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM9TestClassFviEXadL_ZNS4_10add_fieldsEiEEE4callEiEUliE_JiEEEvT_DpT0_(
.param .align 1 .b8 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM9TestClassFviEXadL_ZNS4_10add_fieldsEiEEE4callEiEUliE_JiEEEvT_DpT0__param_0[1],
.param .u32 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM9TestClassFviEXadL_ZNS4_10add_fieldsEiEEE4callEiEUliE_JiEEEvT_DpT0__param_1
)
{
.reg .pred %p<2>;
.reg .b32 %r<13>;
.reg .b64 %rd<6>;


ld.param.u32 %r1, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM9TestClassFviEXadL_ZNS4_10add_fieldsEiEEE4callEiEUliE_JiEEEvT_DpT0__param_1];
mov.u32 %r2, %tid.x;
mov.u32 %r3, %ntid.x;
mov.u32 %r4, %ctaid.x;
mad.lo.s32 %r5, %r3, %r4, %r2;
cvt.s64.s32	%rd1, %r5;
ld.global.u64 %rd2, [__TestClassdata_buffer];
setp.ge.u64	%p1, %rd1, %rd2;
@%p1 bra BB8_2;

cvt.u32.u64	%r6, %rd1;
shl.b64 %rd3, %rd1, 2;
mov.u64 %rd4, __TestClassdata_buffer;
add.s64 %rd5, %rd4, %rd3;
ld.global.u32 %r7, [%rd5+20];
add.s32 %r8, %r7, %r1;
ld.global.u32 %r9, [%rd5+4020];
add.s32 %r10, %r8, %r9;
add.s32 %r11, %r6, %r10;
add.s32 %r12, %r11, 30583;
st.global.u32 [%rd5+20], %r12;

BB8_2:
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


.visible .entry _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I9TestClassLm0EEEJEEEvPT_DpT0_(
.param .u64 _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I9TestClassLm0EEEJEEEvPT_DpT0__param_0
)
{
.reg .pred %p<2>;
.reg .b64 %rd<4>;


ld.param.u64 %rd1, [_ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I9TestClassLm0EEEJEEEvPT_DpT0__param_0];
setp.eq.s64	%p1, %rd1, 0;
@%p1 bra BB10_2;

cvta.to.global.u64 %rd2, %rd1;
mov.u64 %rd3, 0;
st.global.u64 [%rd2], %rd3;
st.global.u64 [%rd2+8], %rd3;

BB10_2:
ret;
}


