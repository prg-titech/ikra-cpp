#ifndef SOA_LAYOUT_H
#define SOA_LAYOUT_H

#include <type_traits>

#include "soa/array_field.h"
#include "soa/class_initialization.h"
#include "soa/constants.h"
#include "soa/cuda.h"
#include "soa/field.h"
#include "soa/partially_inlined_array_field.h"
#include "soa/storage.h"
#include "soa/util.h"

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
// polymorphism or "Curiously Recurring Template Pattern"). UserCapacity is
// the maximum number of instances this class can have. It is a compile-time
// constant to allow for efficient field address computation. AddressMode can
// be either "Zero Addressing Mode" or "Valid Addressing Mode".
// The real capacity of a SOA class is chosen such that `UserCapacity + 1` is
// a multiple of 8 (for alignment reasons).
template<class Self,
         IndexType UserCapacity,
         int AddressMode = kAddressModeZero,
         class StorageStrategy = StaticStorage,
         int LayoutMode = kLayoutModeSoa>
class SoaLayout : SizeNDummy<AddressMode> {
 private:
  // Calculate real capacity of the container, such that its size is a multiple
  // of 8 bytes.
  static constexpr IndexType calculate_capacity() {
    // C++11 constexpr must have only one return statement.
    return ((UserCapacity + 8) / 8) * 8 - 1;
  }

  static const IndexType Capacity = calculate_capacity();

 public:
  using Storage = typename StorageStrategy::template type<Self>;
  static const IndexType kCapacity = Capacity;
  static const bool kIsSoaClass = true;

  // Cannot access members of Storge in here because Storage depends on this
  // class and must not be instantiated yet.
  static constexpr int kStorageMode = storage_mode<StorageStrategy>::value;

  // Check if the compiler supports this address mode.
  static_assert(
      sizeof(AddressingModeCompilerCheck<AddressMode>) == AddressMode,
      "Selected addressing mode not supported by compiler.");

  // AOS layout is only for reference and not fully implemented.
  static_assert(LayoutMode == kLayoutModeSoa || 
      (AddressMode == kAddressModeZero && kStorageMode == kStorageModeStatic),
      "AOS layout allowed only in zero addressing mode with static storage.");

  __ikra_host_device__ static constexpr Storage& storage() {
    return *reinterpret_cast<Storage*>(Self::storage_buffer()); 
  }

  // Define a Field_ alias as a shortcut.
  template<typename T, int Offset>
  using Field = Field_<T, Capacity, Offset, AddressMode,
                       kStorageMode, LayoutMode, Self>;

  // Generate field types. Implement more types as necessary.
  // TODO: Can this be merged with field_type_generator.h?
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(bool)
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(char)
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(double)
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(float)
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(int)
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(uint8_t)
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(uint16_t)
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(uint32_t)
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(uint64_t)

  // This struct serves as a namespace and contains array field types.
  struct array {
    template<typename T, size_t N, int Offset>
    using object = ikra::soa::ArrayObjectField_<
        std::array<T, N>, T, N, Capacity, Offset, AddressMode,
        kStorageMode, LayoutMode, Self>;

    template<typename T, size_t N, int Offset>
    using fully_inlined = ikra::soa::FullyInlinedArrayField_<
        T, N, Capacity, Offset, AddressMode, kStorageMode, LayoutMode, Self>;

    template<typename T, size_t InlineSize, int Offset>
    using partially_inlined = ikra::soa::PartiallyInlinedArrayField_<
        T, InlineSize, Capacity, Offset, AddressMode,
        kStorageMode, LayoutMode, Self>;
  };

  static const int kAddressMode = AddressMode;

  // Create a new instance of this class. Data will be allocated inside
  // storage.data.
  __ikra_device__ void* operator new(size_t count) {
    check_sizeof_class();
    assert(count == sizeof(Self));
    // Check if out of memory.
    assert(size() <= Capacity);
    return get(Self::storage().increase_size(1));
  }

  // Create a new instance of this class with placement new.
  __ikra_device__ void* operator new(size_t /*count*/, void* ptr) {
    // TODO: Add pointer check for various addressing modes.
    check_sizeof_class();
    return ptr;
  }

  // TODO: Implement delete operator.
  __ikra_device__ void operator delete(void* /*ptr*/) {
    assert(false);
  }

  // TODO: Implement delete operator.
  __ikra_device__ void operator delete(void* /*ptr*/, void* /*place*/) {
    assert(false);
  }

  // Create multiple new instances of this class. Data will be allocated inside
  // storage.data.
  __ikra_device__ void* operator new[](size_t count) {
    check_sizeof_class();
    // "count" is the number of new instances.
    return get_uninitialized(Self::storage().increase_size(count/AddressMode));
  }

  // Return the number of instances of this class.
  __ikra_device__ static IndexType size() {
    return Self::storage().size();
  }

  // Return a pointer to an object with a given ID. Do not check if the
  // object was previously initialized.
  __ikra_device__ static Self* get_uninitialized(IndexType id) {
    assert(id <= Capacity);

    // First object has actually ID 1. This is because a nullptr allows the
    // compiler to do special optimizations.
    return get_(id + 1);
  }

#if defined(__CUDA_ARCH__) || !defined(__CUDACC__)
  // Not running in CUDA mode or running on device.
  // Return a pointer to an object with a given ID.
  __ikra_device__ static Self* get(IndexType id) {
    assert(id < Self::storage().size());

    // First object has actually ID 1. This is because a nullptr allows the
    // compiler to do special optimizations.
    return get_(id + 1);
  }
#else
  // Running in CUDA mode and accessing object from host code.
  // Return a pointer to an object with a given ID.
  static Self* get(IndexType id) {
    assert(id <= Capacity);
    assert(id < size());    // Will trigger cudaMemcpy!
    return get_uninitialized(id);
  }
#endif

  // Return a pointer to an object by ID (assuming valid addressing mode).
  // TODO: This method should be private!
  template<int A = AddressMode>
  __ikra_device__
  static typename std::enable_if<A != kAddressModeZero, Self*>::type
  get_(IndexType id) {
    // Start counting from 1 internally.
    assert(id > 0);
    uintptr_t address = reinterpret_cast<uintptr_t>(Self::storage().data_ptr())
                        + id*AddressMode;
    return reinterpret_cast<Self*>(address);
  }

  // Return a pointer to an object by ID (assuming zero addressing mode).
  // TODO: This method should be private!
  template<int A = AddressMode, int L = LayoutMode>
  __ikra_device__
  static typename std::enable_if<A == kAddressModeZero &&
                                 L == kLayoutModeSoa, Self*>::type
  get_(IndexType id) {
    // Start counting from 1 internally.
    assert(id > 0);
    return reinterpret_cast<Self*>(id);
  }

  template<int A = AddressMode, int L = LayoutMode>
  __ikra_device__
  static typename std::enable_if<A == kAddressModeZero &&
                                 L == kLayoutModeAos, Self*>::type
  get_(IndexType id) {
    // Start counting from 1 internally.
    assert(id > 0);
    // Use constant-folded value for address computation.
    constexpr auto cptr_data_offset =
        StorageDataOffset<Storage>::value;
    constexpr auto cptr_storage_buffer = Self::storage_buffer();
    char* buffer_location = reinterpret_cast<char*>(
        cptr_storage_buffer + cptr_data_offset);

    return reinterpret_cast<Self*>(buffer_location
        + id*Self::ObjectSize::value);
  }

  // Return an iterator pointing to the first instance of this class.
  // TODO: CUDA support for iterators.
  static executor::Iterator<Self*> begin() {
    return executor::Iterator<Self*>(Self::get(0));
  }

  // Return an iterator pointing to the last instance of this class + 1.
  static executor::Iterator<Self*> end() {
    return ++executor::Iterator<Self*>(Self::get(size() - 1));
  }

  // Calculate the ID of this object (assuming valid addressing mode).
  template<int A = AddressMode>
  __ikra_device__
  typename std::enable_if<A != kAddressModeZero, IndexType>::type 
  id() const {
    return (reinterpret_cast<uintptr_t>(this) - 
           reinterpret_cast<uintptr_t>(Self::storage().data_ptr())) / A - 1;
  }

  // Calculate the ID of this object (assuming zero addressing mode).
  template<int A = AddressMode, int L = LayoutMode>
  __ikra_device__
  typename std::enable_if<A == kAddressModeZero &&
                          L == kLayoutModeSoa, IndexType>::type 
  id() const {
    return reinterpret_cast<uintptr_t>(this) - 1;
  }

  template<int A = AddressMode, int L = LayoutMode>
  __ikra_device__
  typename std::enable_if<A == kAddressModeZero &&
                          L == kLayoutModeAos, IndexType>::type 
  id() const {
    constexpr auto cptr_data_offset =
        StorageDataOffset<Storage>::value;
    constexpr auto cptr_storage_buffer = Self::storage_buffer();
    uintptr_t buffer_location = reinterpret_cast<uintptr_t>(
        cptr_storage_buffer + cptr_data_offset);
    auto buffer_offset = reinterpret_cast<uintptr_t>(this) - buffer_location;

    assert(buffer_offset % Self::ObjectSize::value == 0);
    return buffer_offset / Self::ObjectSize::value  - 1;
  }

#ifdef __CUDACC__
  template<typename... Args>
  static void initialize_storage(Args... args) {
    storage_cuda_initialize(Self::storage().device_ptr(), args...);
  }
#else
  // Initializes the storage buffer with an actual storage object.
  template<typename... Args>
  static void initialize_storage(Args... args) {
    new (reinterpret_cast<void*>(&Self::storage())) Storage(args...);
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
    static_assert(sizeof(Self) == AddressMode,
                  "SOA class must have only SOA fields.");
  }

  // Compile-time check for the size of this class. This check should fail
  // if this class contains fields that are not declared with the SOA DSL.
  template<int A = AddressMode>
  __ikra_device__
  static typename std::enable_if<A == kAddressModeZero, void>::type
  check_sizeof_class() {
#ifndef __CUDACC__
    static_assert(sizeof(Self) == 0,
                  "SOA class must have only SOA fields.");
#else
    // GPU does not do zero initialization, so it's fine.
#endif  // __CUDACC__
  }
};

#undef IKRA_DEFINE_LAYOUT_FIELD_TYPE

}  // namespace soa
}  // namespace ikra

#endif  // SOA_LAYOUT_H
