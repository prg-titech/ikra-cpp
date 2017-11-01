#ifndef SOA_CUDA_H
#define SOA_CUDA_H

#ifdef __CUDACC__

#ifdef __CUDA_ARCH__
// Make all SOA functions device functions when compiling device code.
#define __ikra_device__ __device__ 
#else
#define __ikra_device__
#endif  // __CUDA_ARCH__

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

namespace ikra {
namespace soa {

// Calls a device lambda function with an arbitrary number of arguments.
template<typename F, typename... Args>
__global__ void kernel_call_lambda(F func, Args... args) {
  func(args...);
}

// This macro expands to a series of statements that execute a given method
// of a given class on a given range of objects. This is done by constructing
// a device lambda that calls the method. A more elegant solution would pass
// a member function pointer to a CUDA kernel that is parameterized by the
// member function type. However, this is unsupported by nvcc and results in
// strange compile errors in the nvcc-generated code.
#define cuda_execute(class_name, method_node, length, ...) \
  /* TODO: Support length > 1024 */ \
  kernel_call_lambda<<<1, length>>>( \
      [] __device__ (auto* base, auto... args) { \
          /* TODO: Assuming zero addressing mode. */ \
          int tid = threadIdx.x + blockIdx.x * blockDim.x; \
          return class_name::get(base->id() + tid)->method_node(args...); \
      }, __VA_ARGS__); cudaDeviceSynchronize();

}  // soa
}  // ikra

#else

#define __ikra_device__

#endif  // __CUDACC__

#endif  // SOA_CUDA_H
