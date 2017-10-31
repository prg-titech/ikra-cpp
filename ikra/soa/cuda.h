#ifndef SOA_CUDA_H
#define SOA_CUDA_H

#ifdef __CUDA_ARCH__
#define __ikra_device__ __host__ __device__ 
#else
#define __ikra_device__
#endif

#endif  // SOA_CUDA_H
