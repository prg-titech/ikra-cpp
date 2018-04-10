#ifndef SOA_STORAGE_H
#define SOA_STORAGE_H

#include <type_traits>

#include "soa/constants.h"
#include "soa/cuda.h"
#include "soa/util.h"

#define IKRA_DEVICE_STORAGE(class_name) \
__device__ alignas(8) char \
    __ ## class_name ## data_buffer[sizeof(class_name::Storage)]; \
constexpr char* class_name::storage_buffer() { \
  return __ ## class_name ## data_buffer; \
}

#define IKRA_HOST_STORAGE(class_name) \
alignas(8) char \
    __ ## class_name ## data_buffer[sizeof(class_name::Storage)]; \
constexpr char* class_name::storage_buffer() { \
  return __ ## class_name ## data_buffer; \
}


namespace ikra {
namespace soa {

using ArenaIndexType = unsigned int;
static const int kStorageModeStatic = 0;
static const int kStorageModeDynamic = 1;

#ifdef __CUDACC__
__device__ IndexType d_previous_storage_size;
#endif  // __CUDACC__

// Storage classes maintain the data of all SOA columns, as well as some meta
// data such as the number of instances of a SOA class. All storage classes are
// singletons, i.e., instantiations of the following classes are tied to a
// specific SOA class.

// Determine the offset of the data field of a storage strategy at compile
// time.
template<typename StorageClass>
struct StorageDataOffset {
  static constexpr uintptr_t value =
#ifdef __clang__
      IKRA_fold(reinterpret_cast<uintptr_t>(
          &reinterpret_cast<StorageClass*>(0)->data_));
#else
      reinterpret_cast<uintptr_t>(&reinterpret_cast<StorageClass*>(0)->data_);
#endif  // __clang__
};

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

 protected:
#ifdef __CUDA_ARCH__
  static __device__ ArenaIndexType atomic_add(ArenaIndexType* addr,
                                              ArenaIndexType increment) {
    return atomicAdd(addr, increment);
  }
#else
  static ArenaIndexType atomic_add(ArenaIndexType* addr,
                                   ArenaIndexType increment) {
    // TODO: Make atomic for thread pool execution
    ArenaIndexType r = *addr;
    *addr += increment;
    return r;
  }
#endif  // __CUDA_ARCH__

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
class alignas(8) DynamicStorage_
    : public StorageStrategySelf<DynamicStorage_<OwnerT, ArenaSize>> {
 public:
  static const int kStorageMode = kStorageModeDynamic;

  using Owner = OwnerT;

  // Allocate data on the heap.
  __ikra_device__ DynamicStorage_() {
    // Note: ObjectSize is accessed within a function here.
    // Add one because first object has ID 0.
    data_ = reinterpret_cast<char*>(
        malloc(Owner::ObjectSize::value * (Owner::kCapacity + 1)));

    if (ArenaSize > 0) {
      arena_head_ = 0;
      arena_base_ = reinterpret_cast<char*>(malloc(ArenaSize));
    } else {
      arena_head_ = 0;
      arena_base_ = nullptr;
    }
  }

  // Use existing data allocation.
  __ikra_device__
  explicit DynamicStorage_(char* ptr, char* arena_ptr = nullptr) {
    assert(ptr != nullptr);
    data_ = ptr;

    if (ArenaSize != 0) {
      assert(arena_ptr != nullptr);
      arena_head_ = 0;
      arena_base_ = arena_ptr;
    }
  }

  // Allocate arena storage, i.e., storage that is not used for a SOA column.
  // For example, inlined dynamic arrays require arena storage for all elements
  // beyond InlineSize.
  __ikra_device__ char* allocate_in_arena(size_t bytes) {
    assert(arena_base_ != nullptr);
    assert(arena_head_ + bytes < ArenaSize);
    auto new_head = StorageStrategyBase::atomic_add(&arena_head_, bytes);
    return arena_base_ + new_head;
  }

  __ikra_device__ char* data_ptr() {
    // Check alignment.
    assert(reinterpret_cast<uintptr_t>(data_) % 8 == 0);
    return data_;
  }

  // Note: Returning as reference not supported in dynamic storage.
  __ikra_device__ char* data_reference() {
    return reinterpret_cast<char*>(data_ptr());
  }

  static const bool kIsStaticStorage = false;

 private:
  template<typename StorageClass>
  friend class StorageDataOffset;

  // The base pointer (start) of the arena.
  char* arena_base_;

  // The current head of the arena (offset in bytes).
  // To be increased after allocation (bump pointer allocation).
  ArenaIndexType arena_head_;

  // A pointer to the memory where SOA columns are stored.
  char* data_;
};

// This class defines the data/memory (char array) for the Owner class.
// Memory is allocated statically, allowing for efficient code generation.
template<class OwnerT, size_t ArenaSize>
class alignas(8) StaticStorage_
    : public StorageStrategySelf<StaticStorage_<OwnerT, ArenaSize>> {
 public:
  static const int kStorageMode = kStorageModeStatic;

  using Owner = OwnerT;

  __ikra_device__ StaticStorage_() {
    arena_head_ = 0;
  }

  __ikra_device__ char* allocate_in_arena(size_t bytes) {
    assert(arena_base_ != nullptr);
    assert(arena_head_ + bytes < ArenaSize);
    auto new_head = StorageStrategyBase::atomic_add(&arena_head_, bytes);
    return arena_base_ + new_head;
  }

  __ikra_device__ char* data_ptr() {
    // Check alignment.
    assert(reinterpret_cast<uintptr_t>(data_) % 8 == 0);
    return &data_[0];
  }

  static const bool kIsStaticStorage = true;

 private:
  template<typename StorageClass>
  friend class StorageDataOffset;

  // Statically allocated data storage for the arena.
  alignas(8) char arena_base_[ArenaSize];

  // Current head pointer of the arena.
  ArenaIndexType arena_head_;

  // Statically allocated data storage for SOA columns.
  // Add one because first object has ID 0.
  alignas(8) char data_[Owner::ObjectSize::value * (Owner::kCapacity + 1)];

 public:
  // Returning the storage buffer data array as reference retains the
  // (type) information that the storage is allocated in global device memory.
  // (When running in device mode.)
  __ikra_device__ auto data_reference() -> decltype(data_)& {
    return data_;
  }
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

// Helper structs for determining the mode of a storage strategy without
// instantiating the template.
template<typename T>
struct storage_mode;

template<>
struct storage_mode<StaticStorage> {
  static const int value = kStorageModeStatic;
};

template<size_t ArenaSize>
struct storage_mode<StaticStorageWithArena<ArenaSize>> {
  static const int value = kStorageModeStatic;
};

template<>
struct storage_mode<DynamicStorage> {
  static const int value = kStorageModeDynamic;
};

template<size_t ArenaSize>
struct storage_mode<DynamicStorageWithArena<ArenaSize>> {
  static const int value = kStorageModeDynamic;
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
