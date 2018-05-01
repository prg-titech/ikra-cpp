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
} \
/* Force instantiation of storage-specific kernel templates. */ \
/* TODO: Does not work if macro is used inside a namespace. */ \
template __global__ void \
    ::ikra::soa::allocate_in_arena_kernel<class_name::Storage>( \
        class_name::Storage*, size_t); \
template __global__ void \
    ::ikra::soa::storage_increase_size_kernel<class_name::Storage>( \
        class_name::Storage*, IndexType);

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
// Helper variables for kernels. Kernel results will be stored in here, so that
// they can be memcopied to the host easily.
__device__ IndexType d_previous_storage_size;
__device__ char* d_arena_allocation;
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

template<typename Self>
class StorageStrategySelf {
 public:
  __ikra_device__ StorageStrategySelf() : size_(0) {}

#ifdef __CUDACC__
  Self* device_ptr() const {
    // Get device address of storage.
    // TODO: This value should be cached.
    Self* d_storage;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_storage), *this);
    assert(cudaPeekAtLastError() == cudaSuccess);
    return d_storage;
  }

  // Convert addresses between device and host.
  template<typename T>
  T* translate_address_host_to_device(T* h_addr) const {
    auto h_data_ptr = reinterpret_cast<uintptr_t>(h_addr);
    auto h_storage_ptr = reinterpret_cast<uintptr_t>(this);
    assert(h_data_ptr >= h_storage_ptr);
    auto data_offset = h_data_ptr - h_storage_ptr;
    auto d_storage_ptr = reinterpret_cast<uintptr_t>(device_ptr());
    return reinterpret_cast<T*>(d_storage_ptr + data_offset);
  }

  template<typename T>
  T* translate_address_device_to_host(T* d_addr) const {
    auto d_data_ptr = reinterpret_cast<uintptr_t>(d_addr);
    auto d_storage_ptr = reinterpret_cast<uintptr_t>(device_ptr());
    assert(d_data_ptr >= d_storage_ptr);
    auto data_offset = d_data_ptr - d_storage_ptr;
    auto h_storage_ptr = reinterpret_cast<uintptr_t>(this);
    return reinterpret_cast<T*>(h_storage_ptr + data_offset);
  }
#endif  // __CUDACC__

#if defined(__CUDA_ARCH__) || !defined(__CUDACC__)
  // Not running in CUDA mode or running on device.
  __ikra_device__ char* allocate_in_arena(size_t bytes) {
    return reinterpret_cast<Self*>(this)->allocate_in_arena_internal(bytes);
  }

  __ikra_device__ IndexType size() const {
    return size_;
  }

  __ikra_device__ IndexType arena_utilization() const {
    return reinterpret_cast<const Self*>(this)->arena_head_;
  }

  using uintptr_t_alt = unsigned long long int;
  __ikra_device__ IndexType increase_size(IndexType increment) {
    // TODO: Find out why this does not work with uintptr_t.
    static_assert(sizeof(uintptr_t_alt) == sizeof(uintptr_t),
        "Internal Error: Size mismatch!");
    uintptr_t_alt* size_ptr = reinterpret_cast<uintptr_t_alt*>(&size_);
    uintptr_t_alt incr = static_cast<uintptr_t_alt>(increment);
    return atomic_add(size_ptr, incr);
  }
#else
  // Running in CUDA mode on host.
  char* allocate_in_arena(size_t bytes);

  IndexType size() const {
    // Copy size to host and return.
    IndexType host_size;
    cudaMemcpy(&host_size, &device_ptr()->size_, sizeof(IndexType),
               cudaMemcpyDeviceToHost);
    assert(cudaPeekAtLastError() == cudaSuccess);
    return host_size;
  }

  __ikra_device__ IndexType arena_utilization() const {
    // Copy utilization to host and return.
    IndexType host_utilization;
    cudaMemcpy(&host_utilization, &device_ptr()->arena_head_,
               sizeof(IndexType), cudaMemcpyDeviceToHost);
    assert(cudaPeekAtLastError() == cudaSuccess);
    return host_utilization;
  }

  // Increase the instance counter on the device.
  // Note: This is a host function.
  IndexType increase_size(IndexType increment);
#endif

 protected:
#ifdef __CUDA_ARCH__
  template<typename T>
  static __device__ T atomic_add(T* addr, T increment) {
    return atomicAdd(addr, increment);
  }
#else
  template<typename T>
  static T atomic_add(T* addr, T increment) {
    // TODO: Make atomic for thread pool execution
    T r = *addr;
    *addr += increment;
    return r;
  }
#endif  // __CUDA_ARCH__

 private:
  // The number of instances of Owner.
  IndexType size_;
};

#if defined(__CUDACC__)
template<typename Storage>
__global__ void storage_increase_size_kernel(Storage* storage,
                                             IndexType increment) {
  d_previous_storage_size = storage->increase_size(increment);
}

template<typename Storage>
__global__ void allocate_in_arena_kernel(Storage* storage, size_t bytes) {
  d_arena_allocation = storage->allocate_in_arena(bytes);
}

#ifndef __CUDA_ARCH__
// Running in host mode. Provide definition of method here because it depends
// on the kernel above.
template<typename Self>
IndexType StorageStrategySelf<Self>::increase_size(IndexType increment) {
  storage_increase_size_kernel<Self><<<1, 1>>>(device_ptr(), increment);
  IndexType h_result_value;
  cudaMemcpyFromSymbol(&h_result_value, d_previous_storage_size,
                       sizeof(h_result_value),
                       0, cudaMemcpyDeviceToHost);
  assert(cudaPeekAtLastError() == cudaSuccess);
  return h_result_value;
}

template<typename Self>
char* StorageStrategySelf<Self>::allocate_in_arena(size_t bytes) {
  allocate_in_arena_kernel<Self><<<1, 1>>>(device_ptr(), bytes);
  assert(cudaPeekAtLastError() == cudaSuccess);
  char* h_result_value;
  cudaMemcpyFromSymbol(&h_result_value, d_arena_allocation,
                       sizeof(char*), 0, cudaMemcpyDeviceToHost);
  assert(cudaPeekAtLastError() == cudaSuccess);

  // Translate to host-relative address.
  return translate_address_device_to_host(h_result_value);
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
 private:
  using SuperT = StorageStrategySelf<DynamicStorage_<OwnerT, ArenaSize>>;
  friend class StorageStrategySelf<DynamicStorage_<OwnerT, ArenaSize>>;

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
  __ikra_device__ char* allocate_in_arena_internal(size_t bytes) {
    assert(arena_base_ != nullptr);
    assert(arena_head_ + bytes < ArenaSize);
    auto new_head = SuperT::atomic_add(&arena_head_,
                                       static_cast<ArenaIndexType>(bytes));
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

 protected:
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
 private:
  using SuperT = StorageStrategySelf<StaticStorage_<OwnerT, ArenaSize>>;
  friend class StorageStrategySelf<StaticStorage_<OwnerT, ArenaSize>>;

 public:
  static const int kStorageMode = kStorageModeStatic;

  using Owner = OwnerT;

  __ikra_device__ StaticStorage_() {
    arena_head_ = 0;
  }

  __ikra_device__ char* allocate_in_arena_internal(size_t bytes) {
    assert(arena_base_ != nullptr);
    assert(ArenaSize == 0 || arena_head_ + bytes < ArenaSize);
    auto new_head = SuperT::atomic_add(&arena_head_,
                                       static_cast<ArenaIndexType>(bytes));
    return arena_base_ + new_head;
  }

  __ikra_device__ char* data_ptr() {
    // Check alignment.
    assert(reinterpret_cast<uintptr_t>(data_) % 8 == 0);
    return &data_[0];
  }

  static const bool kIsStaticStorage = true;

 protected:
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
