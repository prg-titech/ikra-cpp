#ifndef SOA_CUDA_H
#define SOA_CUDA_H

#ifdef __CUDACC__

#define __ikra_device__ __host__ __device__ 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace ikra {
namespace soa {

template<typename F, typename... Args>
__global__ void kernel_call_lambda(F func, Args... args) {
  func(args...);
}

#define cuda_execute(class_name, method_node, length, ...) \
  kernel_call_lambda<<<1, length>>>( \
      [] __device__ (auto* base, auto... args) { \
          /* TODO: Assuming zero addressing mode. */ \
          int tid = threadIdx.x + blockIdx.x * blockDim.x; \
          class_name::get(base->id() + tid)->method_node(args...); \
      }, __VA_ARGS__); cudaDeviceSynchronize();

}  // soa
}  // ikra

#else

#define __ikra_device__

#endif  // __CUDACC__

#endif  // SOA_CUDA_H
