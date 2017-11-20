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
// https://stackoverflow.com/questions/9779105/
// generic-member-function-pointer-as-a-template-parameter
// This struct is used to extract the class name, return type and argument
// types from member function that is passed as a template argument.
template<typename T, T> struct ExecuteKernelProxy;

// Note: The function "func" is a compile-time template parameter.
// This is crucial since taking the address of a device function is forbidden
// in host code when used as an expression.
template<typename T, typename R, typename... Args, R (T::*func)(Args...)>
struct ExecuteKernelProxy<R (T::*)(Args...), func>
{
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
      }
    };
    
    kernel_call_lambda<<<num_blocks, num_threads>>>(
        invoke_function, first, num_objects, args...);
    cudaDeviceSynchronize();
    assert(cudaPeekAtLastError() == cudaSuccess);
  }

  // Invoke CUDA kenrel on all objects.
  static void call(Args... args) {
    IndexType num_objects = T::size();
    IndexType num_blocks = cuda_blocks_1d(num_objects);
    IndexType num_threads = cuda_threads_1d(num_objects);

    auto invoke_function = [] __device__ (Args... args) {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      if (tid < T::size()) {
        auto* object = T::get(tid);
        return (object->*func)(args...);
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

#define IKRA_ALL_BUT_FIRST(first, ...) __VA_ARGS__

// TODO: Reduce in parallel.
// TODO: Rewrite with proxy.
#define cuda_execute_and_reduce(class_name, method_name, \
                                reduce_function, length, ...) \
  /* TODO: Support length > 1024 */ \
  [&] { \
    auto reducer = reduce_function; \
    /* Infer return type of method. Nvcc does not let us use call the */ \
    /* method from within a decltype() token because it is a dev. method. */ \
    /* As a workaround, wrap it within std::bind. */ \
    using return_type = decltype(std::bind(&class_name::method_name, \
        reinterpret_cast<class_name*>(0), IKRA_ALL_BUT_FIRST(__VA_ARGS__))());\
    return_type* d_result; \
    cudaMalloc(&d_result, sizeof(return_type)*(length)); \
    ikra::executor::cuda::kernel_call_lambda_collect_result<<<1, length>>>( \
        [] __device__ (auto* base, auto... args) { \
            /* TODO: Assuming zero addressing mode. */ \
            static_assert(class_name::kAddressMode \
                == ikra::soa::kAddressModeZero, \
                "Not implemented: Valid addressing mode."); \
            int tid = threadIdx.x + blockIdx.x * blockDim.x; \
            return class_name::get(base->id() + tid)->method_name(args...); \
        }, d_result, __VA_ARGS__); \
    return_type* h_result = reinterpret_cast<return_type*>( \
        malloc(sizeof(return_type)*(length))); \
    cudaMemcpy(h_result, d_result, sizeof(return_type)*(length), \
               cudaMemcpyDeviceToHost); \
    return_type reduced_value = h_result[0]; \
    for (uintptr_t i = 1; i < (length); ++i) { \
      reduced_value = reducer(reduced_value, h_result[i]); \
    } \
    cudaFree(d_result); \
    free(h_result); \
    assert(cudaPeekAtLastError() == cudaSuccess); \
    return reduced_value; \
  } ()

}  // cuda
}  // executor
}  // ikra

#endif  // EXECUTOR_CUDA_EXECUTOR_H
