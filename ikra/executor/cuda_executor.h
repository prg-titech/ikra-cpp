#ifndef EXECUTOR_CUDA_EXECUTOR_H
#define EXECUTOR_CUDA_EXECUTOR_H

// Asserts active only in debug mode (NDEBUG).
#include <cassert>
#include <functional>
#include <type_traits>

#include "executor/util.h"
#include "soa/constants.h"
#include "soa/cuda.h"

namespace ikra {
namespace executor {
namespace cuda {

using ikra::soa::IndexType;
using ikra::soa::kAddressModeZero;

template<IndexType VirtualWarpSize>
class KernelConfiguration {
 public:
  KernelConfiguration(IndexType num_blocks, IndexType num_threads)
      : num_blocks_(num_blocks), num_threads_(num_threads) {
    assert(VirtualWarpSize <= num_threads);
  }

  KernelConfiguration(IndexType num_objects)
      : KernelConfiguration<VirtualWarpSize>(cuda_blocks_1d(num_objects),
                                             cuda_threads_1d(num_objects)) {}

  IndexType num_blocks() const { return num_blocks_; }
  IndexType num_threads() const { return num_threads_; }

  static const IndexType kVirtualWarpSize = VirtualWarpSize;

 private:
  const IndexType num_blocks_;
  const IndexType num_threads_;

  static IndexType cuda_blocks_1d(IndexType length) {
    return (length + 256 - 1) / 256;
  }

  static IndexType cuda_threads_1d(IndexType length) {
    return length < 256 ? length : 256;
  }
};

class KernelConfigurationStrategy {
 public:
  // For SFINAE overload selection.
  static const bool kIsConfigurationStrategy = true;
};

// Build a kernel configuration based on the number of objects.
class StandardKernelConfigurationStrategy
    : public KernelConfigurationStrategy {
 public:
  KernelConfiguration<1> build_configuration(IndexType num_threads) const {
    return KernelConfiguration<1>(num_threads);
  }
};

// TODO: Not sure if it's a good idea to define this in a header file.
static const StandardKernelConfigurationStrategy kKernelConfigStandard =
    StandardKernelConfigurationStrategy();

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
__global__ void construct_kernel(IndexType base_id, IndexType num_objects,
                                 Args... args) {
  static_assert(T::kAddressMode == kAddressModeZero,
      "Not implemented: Valid addressing mode.");
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (tid < num_objects) {
    new (T::get_uninitialized(base_id + tid)) T(args...);
  }
}

template<typename T, IndexType VirtualWarpSize, typename... Args>
T* construct(const KernelConfiguration<VirtualWarpSize>& config,
             size_t count, Args... args) {
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

  construct_kernel<T><<<config.num_blocks(), config.num_threads()>>>(
      h_storage_size, count, args...);
  gpuErrchk(cudaDeviceSynchronize());
  assert(cudaPeekAtLastError() == cudaSuccess);

  return h_storage_data_head;
}

template<typename T, typename Config, typename... Args>
typename std::enable_if<Config::kIsConfigurationStrategy, T*>::type
construct(const Config& strategy, size_t count, Args... args) {
  return construct<T>(strategy.build_configuration(count), count, args...);
}

template<typename T, typename... Args>
T* construct(size_t count, Args... args) {
  return construct<T>(kKernelConfigStandard, count, args...);
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
  template<IndexType VirtualWarpSize>
  static void call(const KernelConfiguration<VirtualWarpSize>& config,
                   T* first, IndexType num_objects, Args... args)
  {
    auto invoke_function = [] __device__ (T* first, IndexType num_objects,
                                          Args... args) {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      if (tid < num_objects) {
        auto* object = T::get(first->id() + tid);
        return (object->*func)(args...);
      }
    };
    
    kernel_call_lambda<<<config.num_blocks(), config.num_threads()>>>(
        invoke_function, first, num_objects, args...);
    gpuErrchk(cudaDeviceSynchronize());
    assert(cudaPeekAtLastError() == cudaSuccess);
  }

  // Invoke CUDA kernel on all objects.
  template<IndexType VirtualWarpSize>
  static void call(const KernelConfiguration<VirtualWarpSize>& config,
                   Args... args) {
    call(config, T::get(0), T::size(), args...);
  }

  // Invoke CUDA kernel on all objects. This is an optimized version where the
  // number of objects is a compile-time constant.
  template<IndexType num_objects, IndexType VirtualWarpSize>
  static void call_fixed_size(
      const KernelConfiguration<VirtualWarpSize>& config, Args... args) {
    auto invoke_function = [] __device__ (Args... args) {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      if (tid < num_objects) {
        auto* object = T::get(tid);
        return (object->*func)(args...);
      }
    };

    kernel_call_lambda<<<config.num_blocks(), config.num_threads()>>>(
        invoke_function, args...);
    cudaDeviceSynchronize();
    assert(cudaPeekAtLastError() == cudaSuccess);
  }

  template<typename Config>
  static typename std::enable_if<Config::kIsConfigurationStrategy, void>::type
  call(const Config& strategy, T* first, IndexType num_objects, Args... args) {
    call(strategy.build_configuration(num_objects),
         first, num_objects, args...);
  }

  template<typename Config>
  static typename std::enable_if<Config::kIsConfigurationStrategy, void>::type
  call(const Config& strategy, Args... args) {
    const IndexType num_objects = T::size();
    call(strategy.build_configuration(num_objects), args...);
  }

  template<IndexType num_objects, typename Config>
  static typename std::enable_if<Config::kIsConfigurationStrategy, void>::type
  call(const Config& strategy, Args... args) {
    call_fixed_size(strategy.build_configuration(num_objects), args...);
  }

  static void call(T* first, IndexType num_objects, Args... args) {
    call(kKernelConfigStandard, first, num_objects, args...);
  }

  static void call(Args... args) {
    call(kKernelConfigStandard, args...);
  }

  template<IndexType num_objects>
  static void call_fixed_size(Args... args) {
    call<num_objects>(kKernelConfigStandard, args...);
  }
};

#define cuda_execute(func, ...) \
  ikra::executor::cuda::ExecuteKernelProxy<decltype(func), func> \
      ::call(__VA_ARGS__)

#define cuda_execute_fixed_size(func, size, ...) \
  ikra::executor::cuda::ExecuteKernelProxy<decltype(func), func> \
      ::call_fixed_size<size>(__VA_ARGS__)

template<typename T, T> class ExecuteAndReturnKernelProxy;

template<typename T, typename R, typename... Args, R (T::*func)(Args...)>
class ExecuteAndReturnKernelProxy<R (T::*)(Args...), func>
{
 public:
  // Invoke CUDA kernel. Call method on "num_objects" many objects, starting
  // from object "first".
  template<IndexType VirtualWarpSize>
  static R* call(const KernelConfiguration<VirtualWarpSize>& config, T* first,
                 IndexType num_objects, Args... args)
  {
    // Allocate memory for result of kernel run.
    R* d_result;
    cudaMalloc(&d_result, sizeof(R)*num_objects);

    auto invoke_function = [] __device__ (T* first, IndexType num_objects,
                                          Args... args) {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      if (tid < num_objects) {
        auto* object = T::get(first->id() + tid);
        return (object->*func)(args...);
      }
    };

    kernel_call_lambda_collect_result<<<config.num_blocks(),
                                        config.num_threads()>>>(
        invoke_function, d_result, first, num_objects, args...);
    cudaDeviceSynchronize();
    assert(cudaPeekAtLastError() == cudaSuccess);

    return d_result;
  }

  // Invoke CUDA kernel on all objects.
  template<IndexType VirtualWarpSize>
  static R* call(const KernelConfiguration<VirtualWarpSize>& config,
                 Args... args) {
    return call(config, T::get(0), T::size(), args...);
  }

  template<typename Config>
  static typename std::enable_if<Config::kIsConfigurationStrategy, R*>::type
  call(const Config& strategy, T* first, IndexType num_objects, Args... args) {
    return call(strategy.build_configuration(num_objects),
                first, num_objects, args...);
  }

  template<typename Config>
  static typename std::enable_if<Config::kIsConfigurationStrategy, R*>::type
  call(const Config& strategy, Args... args) {
    const IndexType num_objects = T::size();
    return call(strategy.build_configuration(num_objects), args...);
  }

  static R* call(T* first, IndexType num_objects, Args... args) {
    return call(kKernelConfigStandard, first, num_objects, args...);
  }

  static R* call(Args... args) {
    return call(kKernelConfigStandard, args...);
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
  template<typename Reducer, IndexType VirtualWarpSize>
  static R call(const KernelConfiguration<VirtualWarpSize>& config, T* first,
                IndexType num_objects, Reducer red, Args... args) {
    R* d_result = ExecuteAndReturnKernelProxy<R(T::*)(Args...), func>
        ::call(config, first, num_objects, args...);

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
  template<typename Reducer, IndexType VirtualWarpSize>
  static R call(const KernelConfiguration<VirtualWarpSize>& config,
                Reducer red, Args... args) {
    return call(config, T::get(0), T::size(), red, args...);
  }

  template<typename Reducer, typename Config>
  static typename std::enable_if<Config::kIsConfigurationStrategy, R>::type
  call(const Config& strategy, T* first, IndexType num_objects,
       Reducer red, Args... args) {
    return call(strategy.build_configuration(num_objects),
                red, first, num_objects, args...);
  }

  template<typename Reducer, typename Config>
  static typename std::enable_if<Config::kIsConfigurationStrategy, R>::type
  call(const Config& strategy, Reducer red, Args... args) {
    const IndexType num_objects = T::size();
    return call(strategy.build_configuration(num_objects), red, args...);
  }

  template<typename Reducer>
  static R call(T* first, IndexType num_objects, Reducer red, Args... args) {
    return call(kKernelConfigStandard, first,
                num_objects, red, args...);
  }

  template<typename Reducer>
  static R call(Reducer red, Args... args) {
    return call(kKernelConfigStandard, red, args...);
  }
};

#define cuda_execute_and_reduce(func, ...) \
  ikra::executor::cuda::ExecuteAndReduceKernelProxy<decltype(func), func> \
      ::call(__VA_ARGS__)

}  // cuda
}  // executor
}  // ikra

#endif  // EXECUTOR_CUDA_EXECUTOR_H
