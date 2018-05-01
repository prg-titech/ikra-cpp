#ifndef EXECUTOR_KERNEL_CONFIGURATION_H
#define EXECUTOR_KERNEL_CONFIGURATION_H

// Asserts active only in debug mode (NDEBUG).
#include <cassert>
#include <type_traits>

#include "soa/constants.h"

namespace ikra {
namespace executor {
namespace cuda {

using ikra::soa::IndexType;

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

#define IKRA_extract_virtual_warp_size(first, ...) \
    ikra::executor::cuda::ExtractVirtualWarpSize<decltype(first)>::value()

}  // cuda
}  // executor
}  // ikra

#endif  // EXECUTOR_KERNEL_CONFIGURATION_H
