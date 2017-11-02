#ifndef SOA_LAYOUT_H
#define SOA_LAYOUT_H

#include "soa/array_field.h"
#include "soa/class_initialization.h"
#include "soa/constants.h"
#include "soa/cuda.h"
#include "soa/field.h"
#include "soa/inlined_dynamic_array_field.h"
#include "soa/storage.h"

namespace ikra {
namespace soa {

// This marco is expanded for every primitive data type (such as int, float)
// and generates alias types for SOA field declarations. For example:
// int_<Offset> --> Field<int, Offset>
#define IKRA_DEFINE_LAYOUT_FIELD_TYPE(type) \
  template<int Offset> \
  using soa_ ## type = Field<type, Offset>; \

// sizeof(SizeNDummy) = N;
template<size_t N>
struct SizeNDummy {
  char dummy_[N];
};

// This class is the superclass of user-defined class that should be layouted
// according to SOA (Structure of Arrays). In zero addressing mode, the size
// of this class will be 0. In valid addressing mode, the size of this class
// will be the size of first field.
// Self is the type of the subclass being defined (see also F-bound
// polymorphism or "Curiously Recurring Template Pattern"). Capacity is
// the maximum number of instances this class can have. It is a compile-time
// constant to allow for efficient field address computation. AddressMode can
// be either "Zero Addressing Mode" or "Valid Addressing Mode".
template<class Self,
         IndexType Capacity,
         int AddressMode = kAddressModeZero,
         class StorageStrategy = StaticStorage>
class SoaLayout : SizeNDummy<AddressMode> {
 public:
  using Storage = typename StorageStrategy::template type<Self>;
  const static IndexType kCapacity = Capacity;

  // Define a Field_ alias as a shortcut.
  template<typename T, int Offset>
  using Field = Field_<T, Capacity, Offset, AddressMode, Self>;

  // Generate field types. Implement more types as necessary.
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(bool)
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(char)
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(double)
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(float)
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(int)

  // This struct serves as a namespace and contains array field types.
  struct array {
    template<typename T, size_t N, int Offset>
    using aos = ikra::soa::AosArrayField_<std::array<T, N>, Capacity,
                                          Offset, AddressMode, Self>;

    template<typename T, size_t N, int Offset>
    using soa = ikra::soa::SoaArrayField_<T, N, Capacity,
                                          Offset, AddressMode, Self>;

    template<typename T, size_t InlineSize, int Offset>
    using inline_soa = ikra::soa::SoaInlinedDynamicArrayField_<
        T, InlineSize, Capacity, Offset, AddressMode, Self>;
  };

  static const int kAddressMode = AddressMode;

  // Create a new instance of this class. Data will be allocated inside
  // storage.data.
  __ikra_device__ void* operator new(size_t count) {
    check_sizeof_class();
    assert(count == sizeof(Self));
    // Check if out of memory.
    assert(Self::storage().size <= Capacity);
    return get(Self::storage().size++);
  }

  // TODO: Implement delete operator.
  __ikra_device__ void operator delete(void* /*ptr*/) {
    assert(false);
  }

#ifdef __CUDACC__
  // Create a new instance of this class with placement new. This is only
  // allowed in CUDA mode where we keep track of instance counters in a
  // different way.
  // TODO: Assuming zero addressing mode.
  __device__ void* operator new(size_t count, void* ptr) {
    check_sizeof_class();
    assert(reinterpret_cast<uintptr_t>(ptr) > 0);
    return ptr;
  }

  // TODO: Implement delete operator.
  __device__ void operator delete(void* /*ptr*/, void* /*place*/) {
    assert(false);
  }
#endif  // __CUDACC__

  // Create multiple new instances of this class. Data will be allocated inside
  // storage.data.
  __ikra_device__ void* operator new[](size_t count) {
    check_sizeof_class();
    // Size of this class is 1. "count" is the number of new instances.
    Self* first_ptr = get_uninitialized(Self::storage().size);
    Self::storage().size += count/AddressMode;
    return first_ptr;
  }

  // Return the number of instances of this class.
  __ikra_device__ static IndexType size() {
    return Self::storage().size;
  }

  // Return a pointer to an object with a given ID. Do not check if the
  // object was previously initialized.
  __ikra_device__ static Self* get_uninitialized(IndexType id) {
    assert(id <= Capacity);

    // First object has actually ID 1. This is because a nullptr allows the
    // compiler to do special optimizations.
    return get_(id + 1);
  }

  // Return a pointer to an object with a given ID.
  __ikra_device__ static Self* get(IndexType id) {
    assert(id < Self::storage().size);

    // First object has actually ID 1. This is because a nullptr allows the
    // compiler to do special optimizations.
    return get_(id + 1);
  }

  // Return a pointer to an object by ID (assuming valid addressing mode).
  // TODO: This method should be private!
  template<int A = AddressMode>
  __ikra_device__
  static typename std::enable_if<A != kAddressModeZero, Self*>::type
  get_(IndexType id) {
    // Start counting from 1 internally.
    assert(id > 0);
    uintptr_t address = reinterpret_cast<uintptr_t>(Self::storage().data) +
                        id*AddressMode;
    return reinterpret_cast<Self*>(address);
  }

  // Return a pointer to an object by ID (assuming zero addressing mode).
  // TODO: This method should be private!
  template<int A = AddressMode>
  __ikra_device__
  static typename std::enable_if<A == kAddressModeZero, Self*>::type
  get_(IndexType id) {
    // Start counting from 1 internally.
    assert(id > 0);
    return reinterpret_cast<Self*>(id);
  }

  // Return an iterator pointing to the first instance of this class.
  __ikra_device__ static executor::Iterator<Self*> begin() {
    return executor::Iterator<Self*>(Self::get(0));
  }

  // Return an iterator pointing to the last instance of this class + 1.
  __ikra_device__ static executor::Iterator<Self*> end() {
    return ++executor::Iterator<Self*>(Self::get(size() - 1));
  }

  // Calculate the ID of this object (assuming valid addressing mode).
  template<int A = AddressMode>
  __ikra_device__
  typename std::enable_if<A != kAddressModeZero, IndexType>::type 
  id() const {
    return (reinterpret_cast<uintptr_t>(this) - 
           reinterpret_cast<uintptr_t>(Self::storage().data)) / A - 1;
  }

  // Calculate the ID of this object (assuming zero addressing mode).
  template<int A = AddressMode>
  __ikra_device__
  typename std::enable_if<A == kAddressModeZero, IndexType>::type 
  id() const {
    return reinterpret_cast<uintptr_t>(this) - 1;
  }

#ifdef __CUDACC__
  static Storage* device_storage_pointer() {
    // Get device address of storage.
    Storage* d_storage;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_storage),
                         Self::storage());
    assert(cudaPeekAtLastError() == cudaSuccess);
    return d_storage;
  }

  static void initialize_storage() {
    storage_cuda_initialize(device_storage_pointer());
  }
#else
  // Initializes the storage buffer with an actual storage object.
  static void initialize_storage() {
    new (reinterpret_cast<void*>(&Self::storage())) Storage();
  }
#endif  // __CUDACC__

 private:
  // Compile-time check for the size of this class. This check should fail
  // if this class contains fields that are not declared with the SOA DSL.
  // Assuming valid addressing mode.
  template<int A = AddressMode>
  __ikra_device__
  static typename std::enable_if<A != kAddressModeZero, void>::type
  check_sizeof_class() {
#ifndef __CUDACC__   // TODO: Fix on GPU.
    static_assert(sizeof(Self) == AddressMode,
                  "SOA class must have only SOA fields.");
#endif  // __CUDACC__
  }

  // Compile-time check for the size of this class. This check should fail
  // if this class contains fields that are not declared with the SOA DSL.
  // Assuming zero addressing mode.
  template<int A = AddressMode>
  __ikra_device__
  static typename std::enable_if<A == kAddressModeZero, void>::type
  check_sizeof_class() {
#ifndef __CUDACC__   // TODO: Fix on GPU.
    static_assert(sizeof(Self) == 0,
                  "SOA class must have only SOA fields.");
#endif  // __CUDACC__
  }
};

#undef IKRA_DEFINE_LAYOUT_FIELD_TYPE

}  // namespace soa
}  // namespace ikra

#endif  // SOA_LAYOUT_H
