#ifndef SOA_ARRAY_H
#define SOA_ARRAY_H

#include <type_traits>

#include "soa/constants.h"
#include "soa/field.h"

namespace ikra {
namespace soa {

// Class for field declarations of type array. This class is intended to be
// used with T = std::array and forwards all method invocations to the wrapped
// array object. The array is stored in AoS format.
template<typename T,
         IndexType Capacity,
         uint32_t Offset,
         int AddressMode,
         int StorageMode,
         class Owner>
class AosArrayField_ {
 private:
  using Self = AosArrayField_<T, Capacity, Offset, AddressMode,
                              StorageMode, Owner>;
  static const size_t ArraySize = std::tuple_size<T>::value;

 public:
  static const int kSize = sizeof(T);

using B = typename T::value_type;
#include "soa/addressable_field_shared.inc"

template<size_t Pos>
__ikra_device__ B* array_data_ptr() const {
  return reinterpret_cast<B*>(data_ptr()) + Pos;
}

__ikra_device__ B* array_data_ptr(size_t pos) const {
  return reinterpret_cast<B*>(data_ptr()) + pos;
}

#include "soa/array_shared.inc"
#include "soa/field_shared.inc"
};

// Class for field declarations of type array. T is the base type of the array.
// This class is the SoA counter part of AosArrayField_. Array slots are
// layouted as if they were SoA fields (columns).
template<typename T,
         size_t ArraySize,
         IndexType Capacity,
         uint32_t Offset,
         int AddressMode,
         int StorageMode,
         class Owner>
class SoaArrayField_ {
 private:
  using Self = SoaArrayField_<T, ArraySize, Capacity, Offset,
                              AddressMode, StorageMode, Owner>;

 public:
  static const int kSize = sizeof(std::array<T, ArraySize>);

  // Support calling methods using -> syntax.
  __ikra_device__ const Self* operator->() const {
    return this;
  }

  T* operator&() const  = delete;

  T& get() const = delete;

  operator T&() const = delete;

using B = T;
#include "soa/array_shared.inc"

  // TODO: Implement iterator and other methods.

 protected:
  // Calculate the address of an array element. For details, see comment
  // of data_ptr in Field_.
  template<size_t Pos, int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, T*>::type
  array_data_ptr() const {
    // Ensure that this is a valid pointer: Only those objects may be accessed
    // which were created with the "new" keyword and are thus initialized.
    assert(this->id() < Owner::storage().size());

    auto p_this = reinterpret_cast<uintptr_t>(this);
    auto p_base = Owner::storage().data_reference();
    auto p_result = (p_base + p_this - p_base - A)/A*sizeof(T) +
                    Capacity*(Offset + Pos*sizeof(T));
    return reinterpret_cast<T*>(p_result);
  }

  template<size_t Pos, int A = AddressMode, int S = StorageMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeStatic, T*>::type
  array_data_ptr() const {
    assert(this->id() < Owner::storage().size());

    // Use constant-folded value for address computation.
    constexpr auto cptr_data_offset =
        StorageDataOffset<typename Owner::Storage>::value;
    constexpr auto cptr_storage_buffer = Owner::storage_buffer();
    constexpr auto array_location =
        cptr_storage_buffer + cptr_data_offset +
        Capacity*(Offset + Pos*sizeof(T));

#ifdef __clang__
    // Clang does not allow reinterpret_cast in constexprs.
    constexpr T* soa_array = IKRA_fold(reinterpret_cast<T*>(array_location));
#else
    constexpr T* soa_array = reinterpret_cast<T*>(array_location);
#endif  // __clang__

    return soa_array + reinterpret_cast<uintptr_t>(this);
  }

  template<size_t Pos, int A = AddressMode, int S = StorageMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeDynamic, T*>::type
  array_data_ptr() const {
    assert(this->id() < Owner::storage().size());

    // Cannot constant fold dynamically allocated storage.
    auto p_base = Owner::storage().data_reference();
    return reinterpret_cast<T*>(
        p_base + Capacity*(Offset + Pos*sizeof(T)) +
        reinterpret_cast<uintptr_t>(this)*sizeof(T));
  }

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, T*>::type
  array_data_ptr(size_t pos) const {
    // Ensure that this is a valid pointer: Only those objects may be accessed
    // which were created with the "new" keyword and are thus initialized.
    assert(this->id() < Owner::storage().size());

    auto p_this = reinterpret_cast<uintptr_t>(this);
    auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());
    auto p_base_ref = Owner::storage().data_reference();
    auto p_result = p_base_ref + (p_this - p_base - A)/A*sizeof(T) +
                    Capacity*(Offset + pos*sizeof(T));
    return reinterpret_cast<T*>(p_result);
  }

  template<int A = AddressMode, int S = StorageMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeStatic, T*>::type
  array_data_ptr(size_t pos) const {
    assert(this->id() < Owner::storage().size());

    // Use constant-folded value for address computation.
    constexpr auto cptr_data_offset =
        StorageDataOffset<typename Owner::Storage>::value;
    constexpr auto cptr_storage_buffer = Owner::storage_buffer();
    constexpr auto array_location =
        cptr_storage_buffer + cptr_data_offset + Capacity*Offset;
    T* soa_array = reinterpret_cast<T*>(array_location + pos*sizeof(T)*Capacity);

    return soa_array + reinterpret_cast<uintptr_t>(this);
  }

  template<int A = AddressMode, int S = StorageMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeDynamic, T*>::type
  array_data_ptr(size_t pos) const {
    assert(this->id() < Owner::storage().size());

    // Cannot constant fold dynamically allocated storage.
    auto p_base = Owner::storage().data_reference();
    return reinterpret_cast<T*>(
        p_base + Capacity*(Offset + pos*sizeof(T)) +
        reinterpret_cast<uintptr_t>(this)*sizeof(T));
  }

#include "soa/field_shared.inc"
};

}  // namespace soa
}  // namespace ikra

#endif  // SOA_ARRAY_H
