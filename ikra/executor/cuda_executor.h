#ifndef EXECUTOR_CUDA_EXECUTOR_H
#define EXECUTOR_CUDA_EXECUTOR_H

// Asserts active only in debug mode (NDEBUG).
#include <cassert>

#include "soa/constants.h"

namespace ikra {
namespace executor {
namespace cuda {

using ikra::soa::IndexType;

// TODO: Assuming zero addressing mode.
template<typename T, typename... Args>
__global__ void construct_kernel(IndexType base_id, Args... args) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  new (T::get_(base_id + tid)) T(/*args...*/);
}

template<typename T, typename... Args>
T* construct(size_t count, Args... args) {
  assert(count > 0);

  // TODO: count > 1024
  // TODO: base_id
  construct_kernel<T><<<1, count>>>(0, args...);
  return T::get_(0);
}


}  // cuda
}  // executor
}  // ikra

#endif  // EXECUTOR_CUDA_EXECUTOR_H