#ifndef SOA_STORAGE_H
#define SOA_STORAGE_H

#include "soa/constants.h"
#include "soa/cuda.h"

#define IKRA_DEVICE_STORAGE(class_name) \
__device__ char __ ## class_name ## data_buffer[sizeof(class_name::Storage)]; \
__host__ __device__ class_name::Storage& class_name::storage() { \
  return *reinterpret_cast<Storage*>(__ ## class_name ## data_buffer); \
}

#define IKRA_HOST_STORAGE(class_name) \
char __ ## class_name ## data_buffer[sizeof(class_name::Storage)]; \
class_name::Storage& class_name::storage() { \
  return *reinterpret_cast<Storage*>(__ ## class_name ## data_buffer); \
}


namespace ikra {
namespace soa {

#ifdef __CUDACC__
__device__ IndexType d_previous_storage_size;
#endif  // __CUDACC__

// Storage classes maintain the data of all SOA columns, as well as some meta
// data such as the number of instances of a SOA class. All storage classes are
// singletons, i.e., instantiations of the following classes are tied to a
// specific SOA class.

class StorageStrategyBase {
 public:
  __ikra_device__ StorageStrategyBase() : size_(0) {}

#ifdef __CUDACC__
  StorageStrategyBase* device_ptr() const {
    // Get device address of storage.
    StorageStrategyBase* d_storage;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_storage), *this);
    assert(cudaPeekAtLastError() == cudaSuccess);
    return d_storage;
  }
#endif  // __CUDACC__

#if defined(__CUDA_ARCH__) || !defined(__CUDACC__)
  // Not running in CUDA mode or running on device.
  __ikra_device__ IndexType size() const {
    return size_;
  }

  __ikra_device__ IndexType increase_size(IndexType increment) {
    // TODO: Make this operation atomic.
    IndexType result = size_;
    size_ += increment;
    return result;
  }
#else
  // Running in CUDA mode on host.
  IndexType size() const {
    // Copy size to host and return.
    IndexType host_size;
    cudaMemcpy(&host_size, &device_ptr()->size_, sizeof(IndexType),
               cudaMemcpyDeviceToHost);
    assert(cudaPeekAtLastError() == cudaSuccess);
    return host_size;
  }

  // Increase the instance counter on the device.
  // Note: This is a host function.
  IndexType increase_size(IndexType increment);
#endif

 private:
  // The number of instances of Owner.
  IndexType size_;
};

template<typename Self>
class StorageStrategySelf : public StorageStrategyBase {
 public:
#ifdef __CUDACC__
  Self* device_ptr() const {
    // Get device address of storage.
    Self* d_storage;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_storage), *this);
    assert(cudaPeekAtLastError() == cudaSuccess);
    return d_storage;
  }
#endif  // __CUDACC__
};

#if defined(__CUDACC__)
// Note: This kernel cannot be templatized because template expansion
// for code generation purposes is broken when invoked only through an
// overridden operator new. This is the reason why StorageStrategyBase exists
// in addition to StorageStrategySelf.
__global__ void storage_increase_size_kernel(StorageStrategyBase* storage,
                                             IndexType increment) {
  d_previous_storage_size = storage->increase_size(increment);
}

#ifndef __CUDA_ARCH__
// Running in host mode. Provide definition of method here because it depends
// on the kernel above.
IndexType StorageStrategyBase::increase_size(IndexType increment) {
  storage_increase_size_kernel<<<1, 1>>>(device_ptr(), increment);
  IndexType h_result_value;
  cudaMemcpyFromSymbol(&h_result_value, d_previous_storage_size,
                       sizeof(h_result_value),
                       0, cudaMemcpyDeviceToHost);
  assert(cudaPeekAtLastError() == cudaSuccess);
  return h_result_value;
}
#endif  // __CUDA_ARCH__
#endif  // __CUDACC__

// This class maintains the data (char array) for the Owner class. Memory is
// allocated dynamically. This allows programmers to take control over the
// allocation of the storage buffer. Note: The maximum number of instances of
// a SOA class is still a compile-time constant.
// TODO: Use cudaMalloc for device storage (?).
template<class OwnerT, size_t ArenaSize>
class DynamicStorage_
    : public StorageStrategySelf<DynamicStorage_<OwnerT, ArenaSize>> {
 public:
  using Owner = OwnerT;

  // Allocate data on the heap.
  __ikra_device__ DynamicStorage_() {
    // Note: ObjectSize is accessed within a function here.
    data_ = malloc(Owner::ObjectSize::value * Owner::kCapacity);

    if (ArenaSize > 0) {
      arena_head_ = arena_base_ = malloc(ArenaSize);
    } else {
      arena_head_ = arena_base_ = nullptr;
    }
  }

  // Use existing data allocation.
  __ikra_device__
  explicit DynamicStorage_(void* ptr, void* arena_ptr = nullptr) {
    data_ = ptr;

    if (ArenaSize != 0 && arena_ptr == nullptr) {
      arena_head_ = arena_base_ = malloc(ArenaSize);
    } else {
      arena_head_ = arena_base_ = arena_ptr;
    }
  }

  // Allocate arena storage, i.e., storage that is not used for a SOA column.
  // For example, inlined dynamic arrays require arena storage for all elements
  // beyond InlineSize.
  __ikra_device__ void* allocate_in_arena(size_t bytes) {
    assert(arena_head_ != nullptr);
    void* result = arena_head_;
    arena_head_ = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(arena_head_) + bytes);
    return result;
  }

  __ikra_device__ void* data_ptr() {
    return data_;
  }

 private:
  // The base pointer (start) of the arena.
  void* arena_base_;

  // The current head of the arena. To be increased after allocation (bump
  // pointer allocation).
  void* arena_head_;

  // A pointer to the memory where SOA columns are stored.
  void* data_;
};

// This class defines the data/memory (char array) for the Owner class.
// Memory is allocated statically, allowing for efficient code generation.
template<class OwnerT, size_t ArenaSize>
class StaticStorage_
    : public StorageStrategySelf<StaticStorage_<OwnerT, ArenaSize>> {
 public:
  using Owner = OwnerT;

  __ikra_device__ StaticStorage_() {
    if (ArenaSize == 0) {
      arena_head_ = nullptr;
    } else {
      arena_head_ = arena_base_;
    }
  }

  __ikra_device__ void* allocate_in_arena(size_t bytes) {
    assert(arena_head_ != nullptr);
    // Assert that arena is not full.
    assert(arena_head_ + bytes < arena_base_ + ArenaSize);
    void* result = reinterpret_cast<void*>(arena_head_);
    arena_head_ += bytes;
    return result;
  }

  __ikra_device__ void* data_ptr() {
    return reinterpret_cast<void*>(&data_[0]);
  }

 private:
  // Statically allocated data storage for the arena.
  char arena_base_[ArenaSize];

  // Current head pointer of the arena.
  char* arena_head_;

  // Statically allocated data storage for SOA columns.
  char data_[Owner::ObjectSize::value * Owner::kCapacity];
};

// The following structs are public API types for storage strategies.
// Programmers should not use the "underscore" types directly.
struct DynamicStorage {
  template<class Owner>
  using type = DynamicStorage_<Owner, 0>;
};

template<size_t ArenaSize>
struct DynamicStorageWithArena {
  template<class Owner>
  using type = DynamicStorage_<Owner, ArenaSize>;
};

struct StaticStorage {
  template<class Owner>
  using type = StaticStorage_<Owner, 0>;
};

template<size_t ArenaSize>
struct StaticStorageWithArena {
  template<class Owner>
  using type = StaticStorage_<Owner, ArenaSize>;
};

#ifdef __CUDACC__
template<typename Storage, typename... Args>
__global__ void storage_cuda_initialize_kernel(Storage* buffer, Args... args) {
  // Initialize a storage strategy in a given buffer. See layout.h for
  // initialization of host-side storages.
  // TODO: Check if there's a slowdown here due to zero-initialization of
  // buffer data structure.
  new (reinterpret_cast<void*>(buffer)) Storage(args...);
}

// Initializes a storage container located in "h_storage" with a storage
// strategy "Storage". "h_storage" is the host representation of the
// (statically-allocated) device storage. This function sets the instance
// counter of the class to zero.
template<typename Storage, typename... Args>
void storage_cuda_initialize(Storage* d_storage, Args... args) {
  storage_cuda_initialize_kernel<<<1, 1>>>(d_storage, args...);
  cudaThreadSynchronize();
  assert(cudaPeekAtLastError() == cudaSuccess);
}
#endif  // __CUDACC__

}  // namespace soa
}  // namespace ikra

#endif  // SOA_STORAGE_H
