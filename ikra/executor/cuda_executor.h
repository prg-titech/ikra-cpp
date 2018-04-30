#ifndef EXECUTOR_CUDA_EXECUTOR_H
#define EXECUTOR_CUDA_EXECUTOR_H

// Asserts active only in debug mode (NDEBUG).
#include <cassert>
#include <functional>
#include <type_traits>

#include "executor/util.h"
#include "soa/constants.h"
#include "soa/cuda.h"

// This constant is used to check if cuda_execute is invoked inside of a
// device function. Dev. functions shadow this constant with a template arg.
static const int __Ikra_W_SZ = 0;

namespace ikra {
namespace executor {
namespace cuda {

using ikra::soa::IndexType;
using ikra::soa::kAddressModeZero;

class KernelConfigurationBase {};

template<IndexType VirtualWarpSize>
class KernelConfiguration : public KernelConfigurationBase {
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

  static const bool kIsConfiguration = true;
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

template<IndexType VirtualWarpSize>
class VirtualWarpConfiguration : KernelConfigurationBase {
 public:
  static const bool kIsConfiguration = true;
  static const IndexType kVirtualWarpSize = VirtualWarpSize;
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
  __device__ __host__   // Never called from device, but silences warnings.
  KernelConfiguration<1> build_configuration(IndexType num_threads) const {
    return KernelConfiguration<1>(num_threads);
  }
};

// Contains factories for kernel configurations.
// Outer virtual warp size of 0 indicates host mode.
template<int OuterVirtualWarpSize = 0>
struct KernelConfig {
  template<int O = OuterVirtualWarpSize>
  __device__ static 
  typename std::enable_if<(O > 0), VirtualWarpConfiguration<1>>::type
  standard() {
    return standard_device();
  }

  template<int O = OuterVirtualWarpSize>
  static 
  typename std::enable_if<O == 0, StandardKernelConfigurationStrategy>::type
  standard() {
    return standard_host();
  }

  static StandardKernelConfigurationStrategy standard_host() {
    return StandardKernelConfigurationStrategy();
  }

  __device__ static VirtualWarpConfiguration<1> standard_device() {
    return virtual_warp<1>();
  }

  template<int VirtualWarpSize>
  __device__ static VirtualWarpConfiguration<VirtualWarpSize> virtual_warp() {
    return VirtualWarpConfiguration<VirtualWarpSize>();
  }
};

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

  // Grid-stride processing.
  for (IndexType i = tid; i < num_objects; i += blockDim.x*gridDim.x) {
    new (T::get_uninitialized(base_id + i)) T(args...);
  }
}

template<typename T, typename Config, typename... Args>
typename std::enable_if<Config::kIsConfiguration, T*>::type
construct(const Config& config, size_t count, Args... args) {
  // Constructors with nested parallelism are not possible in this
  // implementation because constructor templates cannot be specified
  // explicitly.
  static_assert(Config::kVirtualWarpSize == 1,
                "Virtual warp size must be 1 in constructors.");
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

template<typename T, typename Strategy, typename... Args>
typename std::enable_if<Strategy::kIsConfigurationStrategy, T*>::type
construct(const Strategy& strategy, size_t count, Args... args) {
  return construct<T>(strategy.build_configuration(count), count, args...);
}

template<typename T, typename... Args>
T* construct(size_t count, Args... args) {
  return construct<T>(KernelConfig<0>::standard(), count, args...);
}

// Calls a device lambda function with an arbitrary number of arguments.
template<typename F, typename... Args>
__global__ void kernel_call_lambda(F func, Args... args) {
  func(args...);
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
  // Running on device. Invoke nested parallelism.
  template<int OuterVirtualWarpSize, typename Config>
  __device__ static
  typename std::enable_if<Config::kIsConfiguration &&
                          (OuterVirtualWarpSize > 0), void>::type
  call(const Config& config, T* first, IndexType num_objects, Args... args) {
    assert(Config::kVirtualWarpSize <= OuterVirtualWarpSize);
    const int tid = threadIdx.x % OuterVirtualWarpSize;
    const IndexType base_id = first->id();

    for (IndexType i = tid;
         i < Config::kVirtualWarpSize*num_objects;
         i += OuterVirtualWarpSize) {
      auto* object = T::get(base_id + i/Config::kVirtualWarpSize);
      (object->*func)(args...);
    }
  }

  // Invoke CUDA kernel. Call method on "num_objects" many objects, starting
  // from object "first".
  // TODO: Figure out how to remove warnings here. (Calling __host__ function
  // from __device__ function.)
  template<int OuterVirtualWarpSize, typename Config>
  static typename std::enable_if<Config::kIsConfiguration &&
                                 OuterVirtualWarpSize == 0, void>::type
  call(const Config& config, T* first, IndexType num_objects, Args... args) {
    auto invoke_function = [] __device__ (T* first, IndexType num_objects,
                                          Args... args) {
      const int tid = threadIdx.x + blockIdx.x * blockDim.x;
      const IndexType base_id = first->id();

      // Grid-stride processing.
      for (IndexType i = tid;
           i < Config::kVirtualWarpSize*num_objects;
           i += blockDim.x*gridDim.x) {
        auto* object = T::get(base_id + i/Config::kVirtualWarpSize);
        (object->*func)(args...);
      }
    };
    
    kernel_call_lambda<<<config.num_blocks(), config.num_threads()>>>(
        invoke_function, first, num_objects, args...);
    gpuErrchk(cudaDeviceSynchronize());
    assert(cudaPeekAtLastError() == cudaSuccess);
  }

  // Invoke CUDA kernel on all objects.
  template<int OuterVirtualWarpSize, typename Config>
  __device__ __host__ static
  typename std::enable_if<Config::kIsConfiguration, void>::type
  call(const Config& config, Args... args) {
    call<OuterVirtualWarpSize>(config, T::get(0), T::size(), args...);
  }

  template<int OuterVirtualWarpSize, typename Strategy>
  __device__ __host__ static
  typename std::enable_if<Strategy::kIsConfigurationStrategy, void>::type
  call(const Strategy& strategy, T* first, IndexType num_objects,
       Args... args) {
    call<OuterVirtualWarpSize>(strategy.build_configuration(num_objects),
                               first, num_objects, args...);
  }

  template<int OuterVirtualWarpSize, typename Strategy>
  __device__ __host__ static
  typename std::enable_if<Strategy::kIsConfigurationStrategy, void>::type
  call(const Strategy& strategy, Args... args) {
    const IndexType num_objects = T::size();
    call<OuterVirtualWarpSize>(strategy.build_configuration(num_objects),
                               args...);
  }

  template<int OuterVirtualWarpSize>
  static
  void call(T* first, IndexType num_objects, Args... args) {
    call<OuterVirtualWarpSize>(KernelConfig<OuterVirtualWarpSize>::standard(),
                               first, num_objects, args...);
  }

  template<int OuterVirtualWarpSize>
  static void call(Args... args) {
    call<OuterVirtualWarpSize>(KernelConfig<OuterVirtualWarpSize>::standard(),
                               args...);
  }

  // Invoke CUDA kernel on all objects. This is an optimized version where the
  // number of objects is a compile-time constant.
  // Running on device. Invoke nested parallelism.
  template<int OuterVirtualWarpSize, IndexType num_objects, typename Config>
  __device__ static
  typename std::enable_if<Config::kIsConfiguration &&
                          (OuterVirtualWarpSize > 0), void>::type
  call_fixed_size(const Config& config, T* first, Args... args) {
    assert(Config::kVirtualWarpSize <= OuterVirtualWarpSize);
    const int tid = threadIdx.x % OuterVirtualWarpSize;
    const IndexType base_id = first->id();

    for (IndexType i = tid;
         i < Config::kVirtualWarpSize*num_objects;
         i += OuterVirtualWarpSize) {
      auto* object = T::get(base_id + i/Config::kVirtualWarpSize);
      (object->*func)(args...);
    }
  }

  // Invoke CUDA kernel. Call method on "num_objects" many objects, starting
  // from object "first".
  // TODO: Figure out how to remove warnings here. (Calling __host__ function
  // from __device__ function.)
  template<int OuterVirtualWarpSize, IndexType num_objects, typename Config>
  static typename std::enable_if<Config::kIsConfiguration &&
                                 OuterVirtualWarpSize == 0, void>::type
  call_fixed_size(const Config& config, T* first, Args... args) {
    auto invoke_function = [] __device__ (T* first, IndexType num_objects,
                                          Args... args) {
      const int tid = threadIdx.x + blockIdx.x * blockDim.x;
      const IndexType base_id = first->id();

      // Grid-stride processing.
      for (IndexType i = tid;
           i < Config::kVirtualWarpSize*num_objects;
           i += blockDim.x*gridDim.x) {
        auto* object = T::get(base_id + i/Config::kVirtualWarpSize);
        (object->*func)(args...);
      }
    };
    
    kernel_call_lambda<<<config.num_blocks(), config.num_threads()>>>(
        invoke_function, first, num_objects, args...);
    gpuErrchk(cudaDeviceSynchronize());
    assert(cudaPeekAtLastError() == cudaSuccess);
  }

  template<int OuterVirtualWarpSize, IndexType num_objects, typename Strategy>
  static typename std::enable_if<Strategy::kIsConfigurationStrategy,
                                 void>::type
  call_fixed_size(const Strategy& strategy, Args... args) {
    call_fixed_size<OuterVirtualWarpSize, num_objects>(
        strategy.build_configuration(num_objects), args...);
  }

  template<int OuterVirtualWarpSize, IndexType num_objects>
  static void call_fixed_size(Args... args) {
    call<OuterVirtualWarpSize, num_objects>(
        KernelConfig<OuterVirtualWarpSize>::standard(), args...);
  }
};

// The constexpr function `value` will extract the virtual warp size from a
// kernel configuration or kernel configuration strategy. For other types, it
// will return the default value 1.
template<typename T = int>
struct ExtractVirtualWarpSize {
  template<typename U = T>
  static constexpr
  typename std::enable_if<std::is_base_of<KernelConfigurationBase, U>::value,
                          IndexType>::type value() {
    return U::kVirtualWarpSize;
  }

  template<typename U = T>
  static constexpr
  typename std::enable_if<
      std::is_base_of<KernelConfigurationStrategy, U>::value,
                      IndexType>::type value() {
    return std::result_of<decltype(&U::build_configuration)(T, IndexType)>
        ::type::kVirtualWarpSize;
  }

  // Default case. No virtual warp size specified.
  template<typename U = T>
  static constexpr typename std::enable_if<
      !std::is_base_of<KernelConfigurationStrategy, U>::value &&
      !std::is_base_of<KernelConfigurationBase, U>::value,
          IndexType>::type value() {
    return 1;
  }
};

#define extract_virtual_warp_size(first, ...) \
    ikra::executor::cuda::ExtractVirtualWarpSize<decltype(first)>::value()

// If running on device, utilize nested parallelism.
#define cuda_execute_vw(func, ...) \
  ikra::executor::cuda::ExecuteKernelProxy< \
      decltype(func<extract_virtual_warp_size(__VA_ARGS__)>), \
      func<extract_virtual_warp_size(__VA_ARGS__)>> \
          ::call<__Ikra_W_SZ>(__VA_ARGS__)

#define cuda_execute(func, ...) \
  ikra::executor::cuda::ExecuteKernelProxy<decltype(func), func> \
      ::call<__Ikra_W_SZ>(__VA_ARGS__)

#define cuda_execute_fixed_size(func, size, ...) \
  ikra::executor::cuda::ExecuteKernelProxy< \
      decltype(func<extract_virtual_warp_size(__VA_ARGS__)>), \
      func<extract_virtual_warp_size(__VA_ARGS__)>> \
          ::call_fixed_size<__Ikra_W_SZ, size>(__VA_ARGS__)

template<typename T, T> class ExecuteAndReturnKernelProxy;

template<typename T, typename R, typename... Args, R (T::*func)(Args...)>
class ExecuteAndReturnKernelProxy<R (T::*)(Args...), func>
{
 public:
  // Invoke CUDA kernel. Call method on "num_objects" many objects, starting
  // from object "first".
  template<int OuterVirtualWarpSize, typename Config>
  static typename std::enable_if<Config::kIsConfiguration &&
                                 OuterVirtualWarpSize == 0, R*>::type
  call(const Config& config, T* first, IndexType num_objects, Args... args) {
    // Allocate memory for result of kernel run.
    R* d_result;
    cudaMalloc(&d_result, sizeof(R)*num_objects);

    auto invoke_function = [] __device__ (R* result, T* first,
                                          IndexType num_objects,
                                          Args... args) {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      IndexType base_id = first->id();

      // Grid-stride processing.
      for (IndexType i = tid;
           i < Config::kVirtualWarpSize*num_objects;
           i += blockDim.x*gridDim.x) {
        auto* object = T::get(base_id + i/Config::kVirtualWarpSize);
        result[i] = (object->*func)(args...);
      }
    };

    kernel_call_lambda<<<config.num_blocks(), config.num_threads()>>>(
        invoke_function, d_result, first, num_objects, args...);
    cudaDeviceSynchronize();
    assert(cudaPeekAtLastError() == cudaSuccess);

    return d_result;
  }

  // Invoke CUDA kernel on all objects.
  template<int OuterVirtualWarpSize, typename Config>
  static typename std::enable_if<Config::kIsConfiguration, R*>::type
  call(const Config& config, Args... args) {
    return call<OuterVirtualWarpSize>(config, T::get(0), T::size(), args...);
  }

  template<int OuterVirtualWarpSize, typename Strategy>
  static typename std::enable_if<Strategy::kIsConfigurationStrategy, R*>::type
  call(const Strategy& strategy, T* first, IndexType num_objects,
       Args... args) {
    return call<OuterVirtualWarpSize>(
        strategy.build_configuration(num_objects),
        first, num_objects, args...);
  }

  template<int OuterVirtualWarpSize, typename Strategy>
  static typename std::enable_if<Strategy::kIsConfigurationStrategy, R*>::type
  call(const Strategy& strategy, Args... args) {
    const IndexType num_objects = T::size();
    return call<OuterVirtualWarpSize>(
        strategy.build_configuration(num_objects), args...);
  }

  template<int OuterVirtualWarpSize>
  static R* call(T* first, IndexType num_objects, Args... args) {
    return call<OuterVirtualWarpSize>(
        KernelConfig<OuterVirtualWarpSize>::standard(), first,
        num_objects, args...);
  }

  template<int OuterVirtualWarpSize>
  static R* call(Args... args) {
    return call<OuterVirtualWarpSize>(
        KernelConfig<OuterVirtualWarpSize>::standard(), args...);
  }
};

#define cuda_execute_and_return(func, ...) \
  ikra::executor::cuda::ExecuteAndReturnKernelProxy<decltype(func), func> \
      ::call<__Ikra_W_SZ>(__VA_ARGS__)


template<typename T, T> class ExecuteAndReduceKernelProxy;

template<typename T, typename R, typename... Args, R (T::*func)(Args...)>
class ExecuteAndReduceKernelProxy<R (T::*)(Args...), func>
{
 public:
  // Invoke CUDA kernel. Call method on "num_objects" many objects, starting
  // from object "first".
  template<int OuterVirtualWarpSize, typename Reducer, typename Config>
  static typename std::enable_if<Config::kIsConfiguration &&
                                 OuterVirtualWarpSize == 0, R>::type
  call(const Config& config, T* first,
       IndexType num_objects, Reducer red, Args... args) {
    R* d_result = ExecuteAndReturnKernelProxy<R(T::*)(Args...), func>
        ::template call<OuterVirtualWarpSize>(config, first,
                                              num_objects, args...);

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
  template<int OuterVirtualWarpSize, typename Reducer, typename Config>
  static typename std::enable_if<Config::kIsConfiguration, R>::type
  call(const Config& config, Reducer red, Args... args) {
    return call<OuterVirtualWarpSize>(config, T::get(0), T::size(),
                                      red, args...);
  }

  template<int OuterVirtualWarpSize, typename Reducer, typename Strategy>
  static typename std::enable_if<Strategy::kIsConfigurationStrategy, R>::type
  call(const Strategy& strategy, T* first, IndexType num_objects,
       Reducer red, Args... args) {
    return call<OuterVirtualWarpSize>(
        strategy.build_configuration(num_objects), red, first,
        num_objects, args...);
  }

  template<int OuterVirtualWarpSize, typename Reducer, typename Strategy>
  static typename std::enable_if<Strategy::kIsConfigurationStrategy, R>::type
  call(const Strategy& strategy, Reducer red, Args... args) {
    const IndexType num_objects = T::size();
    return call<OuterVirtualWarpSize>(
        strategy.build_configuration(num_objects), red, args...);
  }

  template<int OuterVirtualWarpSize, typename Reducer>
  static R call(T* first, IndexType num_objects, Reducer red, Args... args) {
    return call<OuterVirtualWarpSize>(
        KernelConfig<OuterVirtualWarpSize>::standard(), first,
        num_objects, red, args...);
  }

  template<int OuterVirtualWarpSize, typename Reducer>
  static R call(Reducer red, Args... args) {
    return call<OuterVirtualWarpSize>(
        KernelConfig<OuterVirtualWarpSize>::standard(), red, args...);
  }
};

#define cuda_execute_and_reduce(func, ...) \
  ikra::executor::cuda::ExecuteAndReduceKernelProxy<decltype(func), func> \
      ::call<__Ikra_W_SZ>(__VA_ARGS__)

}  // cuda
}  // executor
}  // ikra

#endif  // EXECUTOR_CUDA_EXECUTOR_H
