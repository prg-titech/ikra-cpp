#ifndef EXECUTOR_CUDA_EXECUTOR_H
#define EXECUTOR_CUDA_EXECUTOR_H

#if __cplusplus < 201402L
  // Use CUDA Toolkit 9.0 or higher.
  #error GPU support requires at least a C++14 compliant compiler.
#endif

// Asserts active only in debug mode (NDEBUG).
#include <cassert>
#include <functional>

#include "executor/util.h"
#include "soa/constants.h"
#include "soa/cuda.h"

namespace ikra {
namespace executor {
namespace cuda {

using ikra::soa::IndexType;
using ikra::soa::kAddressModeZero;

// Helper variables for easier data transfer.
// Storage size (number of instances).
__device__ IndexType d_storage_size;
// Storage header pointer, i.e., points to the location where the next object
// will be placed.
__device__ void* d_storage_data_head;

// This kernel must not be run with more than one thread.
template<typename T>
__global__ void increment_class_size_kernel(IndexType increment) {
  d_storage_data_head = reinterpret_cast<void*>(
      T::get_uninitialized(T::storage().size()));
  d_storage_size = T::storage().size();
  T::storage().increase_size(increment);
}

// TODO: Assuming zero addressing mode.
template<typename T, typename... Args>
__global__ void construct_kernel(IndexType base_id, Args... args) {
  static_assert(T::kAddressMode == kAddressModeZero,
      "Not implemented: Valid addressing mode.");
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // Use placement new to avoid atomic increment of ID.
  new (T::get_uninitialized(base_id + tid)) T(args...);
}

template<typename T, typename... Args>
T* construct(size_t count, Args... args) {
  assert(count > 0);

  // Increment class size. "h_storage_size" is the size (number of instances)
  // before instance construction.
  increment_class_size_kernel<T><<<1, 1>>>(count);
  IndexType h_storage_size;
  cudaMemcpyFromSymbol(&h_storage_size, d_storage_size, sizeof(h_storage_size),
                       0, cudaMemcpyDeviceToHost);
  T* h_storage_data_head;
  cudaMemcpyFromSymbol(&h_storage_data_head, d_storage_data_head,
                       sizeof(h_storage_data_head), 0, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();

  // TODO: Support more than 1024 elements.
  construct_kernel<T><<<1, count>>>(h_storage_size, args...);
  cudaThreadSynchronize();

  return h_storage_data_head;
}

// Calls a device lambda function with an arbitrary number of arguments.
template<typename F, typename... Args>
__global__ void kernel_call_lambda(F func, Args... args) {
  func(args...);
}

template<typename F, typename R, typename... Args>
__global__ void kernel_call_lambda_collect_result(
    F func, R* result, Args... args) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  result[tid] = func(args...);
}

IndexType cuda_blocks_1d(IndexType length) {
  return (length + 256 - 1) / 256;
}

IndexType cuda_threads_1d(IndexType length) {
  return length < 256 ? length : 256;
}

// Proxy idea based on:
// https://stackoverflow.com/questions/9779105/generic-member-function-pointer-as-a-template-parameter
// This class is used to extract the class name, return type and argument
// types from member function that is passed as a template argument.
template<typename T, T> class ExecuteKernelProxy;

// Note: The function "func" is a compile-time template parameter.
// This is crucial since taking the address of a device function is forbidden
// in host code when used as an expression.
template<typename T, typename R, typename... Args, R (T::*func)(Args...)>
class ExecuteKernelProxy<R (T::*)(Args...), func>
{
 public:
  // Invoke CUDA kernel. Call method on "num_objects" many objects, starting
  // from object "first".
  static void call(T* first, IndexType num_objects, Args... args)
  {
    IndexType num_blocks = cuda_blocks_1d(num_objects);
    IndexType num_threads = cuda_threads_1d(num_objects);

    auto invoke_function = [] __device__ (T* first, IndexType num_objects,
                                          Args... args) {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      if (tid < num_objects) {
        auto* object = T::get(first->id() + tid);
        return (object->*func)(args...);
      } else {
        // This branch is never reached.
        assert(false);
      }
    };
    
    kernel_call_lambda<<<num_blocks, num_threads>>>(
        invoke_function, first, num_objects, args...);
    cudaDeviceSynchronize();
    assert(cudaPeekAtLastError() == cudaSuccess);
  }

  // Invoke CUDA kernel on all objects.
  static void call(Args... args) {
    IndexType num_objects = T::size();
    IndexType num_blocks = cuda_blocks_1d(num_objects);
    IndexType num_threads = cuda_threads_1d(num_objects);

    auto invoke_function = [] __device__ (Args... args) {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      if (tid < T::size()) {
        auto* object = T::get(tid);
        return (object->*func)(args...);
      } else {
        // This branch is never reached.
        assert(false);
      }
    };

    kernel_call_lambda<<<num_blocks, num_threads>>>(invoke_function, args...);
    cudaDeviceSynchronize();
    assert(cudaPeekAtLastError() == cudaSuccess);
  }
};

#define cuda_execute(func, ...) \
  ikra::executor::cuda::ExecuteKernelProxy<decltype(func), func> \
      ::call(__VA_ARGS__)


template<typename T, T> class ExecuteAndReturnKernelProxy;

template<typename T, typename R, typename... Args, R (T::*func)(Args...)>
class ExecuteAndReturnKernelProxy<R (T::*)(Args...), func>
{
 public:
  // Invoke CUDA kernel. Call method on "num_objects" many objects, starting
  // from object "first".
  static R* call(T* first, IndexType num_objects, Args... args)
  {
    // Allocate memory for result of kernel run.
    R* d_result;
    cudaMalloc(&d_result, sizeof(R)*num_objects);

    // Kernel configuration.
    IndexType num_blocks = cuda_blocks_1d(num_objects);
    IndexType num_threads = cuda_threads_1d(num_objects);

    auto invoke_function = [] __device__ (T* first, IndexType num_objects,
                                          Args... args) {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      if (tid < num_objects) {
        auto* object = T::get(first->id() + tid);
        return (object->*func)(args...);
      } else {
        // This branch is never reached.
        assert(false);
        return (T::get(0)->*func)(args...);
      }
    };

    kernel_call_lambda_collect_result<<<num_blocks, num_threads>>>(
        invoke_function, d_result, first, num_objects, args...);
    cudaDeviceSynchronize();
    assert(cudaPeekAtLastError() == cudaSuccess);

    return d_result;
  }

  // Invoke CUDA kernel on all objects.
  static R* call(Args... args) {
    // Kernel configuration.
    IndexType num_objects = T::size();
    IndexType num_blocks = cuda_blocks_1d(num_objects);
    IndexType num_threads = cuda_threads_1d(num_objects);

    // Allocate memory for result of kernel run.
    R* d_result;
    cudaMalloc(&d_result, sizeof(R)*num_objects);

    auto invoke_function = [] __device__ (Args... args) {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      if (tid < T::size()) {
        auto* object = T::get(tid);
        return (object->*func)(args...);
      } else {
        // This branch is never reached.
        assert(false);
        return (T::get(0)->*func)(args...);
      }
    };

    kernel_call_lambda_collect_result<<<num_blocks, num_threads>>>(
        invoke_function, d_result, args...);
    cudaDeviceSynchronize();
    assert(cudaPeekAtLastError() == cudaSuccess);

    return d_result;
  }
};

#define cuda_execute_and_return(func, ...) \
  ikra::executor::cuda::ExecuteAndReturnKernelProxy<decltype(func), func> \
      ::call(__VA_ARGS__)


template<typename T, T> class ExecuteAndReduceKernelProxy;

template<typename T, typename R, typename... Args, R (T::*func)(Args...)>
class ExecuteAndReduceKernelProxy<R (T::*)(Args...), func>
{
 public:
  // Invoke CUDA kernel. Call method on "num_objects" many objects, starting
  // from object "first".
  template<typename Reducer>
  static R call(T* first, IndexType num_objects, Reducer red, Args... args) {
    R* d_result = ExecuteAndReturnKernelProxy<R(T::*)(Args...), func>
        ::call(first, num_objects, args...);

    R* h_result = reinterpret_cast<R*>(malloc(sizeof(R)*num_objects));
    cudaMemcpy(h_result, d_result, sizeof(R)*num_objects,
               cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    assert(cudaPeekAtLastError() == cudaSuccess);

    // TODO: Not sure why but Reducer does work when code is moved to a
    // separate method.
    R accumulator = h_result[0];
    for (IndexType i = 1; i < num_objects; ++i) {
      accumulator = red(accumulator, h_result[i]);
    }

    free(h_result);
    return accumulator;
  }

  // Invoke CUDA kernel on all objects.
  template<typename Reducer>
  static R call(Reducer red, Args... args) {
    R* d_result = ExecuteAndReturnKernelProxy<R(T::*)(Args...), func>
        ::call(args...);
    IndexType num_objects = T::size();

    R* h_result = reinterpret_cast<R*>(malloc(sizeof(R)*num_objects));
    cudaMemcpy(h_result, d_result, sizeof(R)*num_objects,
               cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    assert(cudaPeekAtLastError() == cudaSuccess);

    // TODO: Not sure why but Reducer does work when code is moved to a
    // separate method.
    R accumulator = h_result[0];
    for (IndexType i = 1; i < num_objects; ++i) {
      accumulator = red(accumulator, h_result[i]);
    }

    free(h_result);
    return accumulator;
  }
};

#define cuda_execute_and_reduce(func, ...) \
  ikra::executor::cuda::ExecuteAndReduceKernelProxy<decltype(func), func> \
      ::call(__VA_ARGS__)

}  // cuda
}  // executor
}  // ikra

#endif  // EXECUTOR_CUDA_EXECUTOR_H
