#ifndef SOA_CUDA_H
#define SOA_CUDA_H

#ifdef __CUDACC__

#ifdef __CUDA_ARCH__
// Make all SOA functions device functions when compiling device code.
#define __ikra_device__ __device__ 
#else
#define __ikra_device__
#endif  // __CUDA_ARCH__

#define __ikra_host_device__ __host__ __device__

// Error checking code taken from:
// https://stackoverflow.com/questions/14038589/
// what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n",
              cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#else

#define __ikra_device__
#define __ikra_host_device__

#endif  // __CUDACC__

#endif  // SOA_CUDA_H
