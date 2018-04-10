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
         typename B,
         size_t ArraySize,
         IndexType Capacity,
         uint32_t Offset,
         int AddressMode,
         int StorageMode,
         class Owner>
class AosArrayField_ {
 private:
  using Self = AosArrayField_<T, B, ArraySize, Capacity, Offset, AddressMode,
                              StorageMode, Owner>;
 public:
  static const uint32_t DBG_OFFSET = Offset;  // For debugging.

  static const int kSize = sizeof(T);
  static_assert(kSize == sizeof(B)*ArraySize,
                "Internal error: Array size mismatch.");

  template<size_t Pos>
  __ikra_device__ B* array_data_ptr() const {
    assert(this->id() < Owner::storage().size());
    static_assert(Pos < ArraySize, "Array index out of bounds.");
    return reinterpret_cast<B*>(data_ptr()) + Pos;
  }

  __ikra_device__ B* array_data_ptr(size_t pos) const {
    assert(this->id() < Owner::storage().size());
    assert(pos < ArraySize);
    return reinterpret_cast<B*>(data_ptr()) + pos;
  }

#include "soa/addressable_field_shared.inc"
#include "soa/array_shared.inc"
#include "soa/field_shared.inc"
};

// Class for field declarations of type array. B is the base type of the array.
// This class is the SoA counter part of AosArrayField_. Array slots are
// layouted as if they were SoA fields (columns).
template<typename B,
         size_t ArraySize,
         IndexType Capacity,
         uint32_t Offset,
         int AddressMode,
         int StorageMode,
         class Owner>
class SoaArrayField_ {
 private:
  using Self = SoaArrayField_<B, ArraySize, Capacity, Offset,
                              AddressMode, StorageMode, Owner>;

 public:
  static const uint32_t DBG_OFFSET = Offset;  // For debugging.

  static const int kSize = sizeof(B)*ArraySize;

  // Support calling methods using -> syntax.
  __ikra_device__ const Self* operator->() const {
    return this;
  }

  B* operator&() const  = delete;

  B& get() const = delete;

  operator B&() const = delete;

  // TODO: Implement iterator and other methods.

 protected:
  // Calculate the address of an array element. For details, see comment
  // of data_ptr in Field_.
  template<size_t Pos, int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, B*>::type
  array_data_ptr() const {
    static_assert(Pos < ArraySize, "Array index out of bounds.");
    // Ensure that this is a valid pointer: Only those objects may be accessed
    // which were created with the "new" keyword and are thus initialized.
    assert(this->id() < Owner::storage().size());

    auto p_this = reinterpret_cast<uintptr_t>(this);
    auto p_base = Owner::storage().data_reference();
    auto p_result = (p_base + p_this - p_base - A)/A*sizeof(B) +
                    (Capacity+1)*(Offset + Pos*sizeof(B));
    return reinterpret_cast<B*>(p_result);
  }

  template<size_t Pos, int A = AddressMode, int S = StorageMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeStatic, B*>::type
  array_data_ptr() const {
    static_assert(Pos < ArraySize, "Array index out of bounds.");
    assert(this->id() < Owner::storage().size());

    // Use constant-folded value for address computation.
    constexpr auto cptr_data_offset =
        StorageDataOffset<typename Owner::Storage>::value;
    constexpr auto cptr_storage_buffer = Owner::storage_buffer();
    constexpr auto array_location =
        cptr_storage_buffer + cptr_data_offset +
        (Capacity+1)*(Offset + Pos*sizeof(B));

#ifdef __clang__
    // Clang does not allow reinterpret_cast in constexprs.
    constexpr B* soa_array = IKRA_fold(reinterpret_cast<B*>(array_location));
#else
    constexpr B* soa_array = reinterpret_cast<B*>(array_location);
#endif  // __clang__

    return soa_array + reinterpret_cast<uintptr_t>(this);
  }

  template<size_t Pos, int A = AddressMode, int S = StorageMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeDynamic, B*>::type
  array_data_ptr() const {
    static_assert(Pos < ArraySize, "Array index out of bounds.");
    assert(this->id() < Owner::storage().size());

    // Cannot constant fold dynamically allocated storage.
    auto p_base = Owner::storage().data_reference();
    return reinterpret_cast<B*>(
        p_base + (Capacity+1)*(Offset + Pos*sizeof(B)) +
        reinterpret_cast<uintptr_t>(this)*sizeof(B));
  }

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, B*>::type
  array_data_ptr(size_t pos) const {
    assert(pos < ArraySize);
    // Ensure that this is a valid pointer: Only those objects may be accessed
    // which were created with the "new" keyword and are thus initialized.
    assert(this->id() < Owner::storage().size());

    auto p_this = reinterpret_cast<uintptr_t>(this);
    auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());
    auto p_base_ref = Owner::storage().data_reference();
    auto p_result = p_base_ref + (p_this - p_base - A)/A*sizeof(B) +
                    (Capacity+1)*(Offset + pos*sizeof(B));
    return reinterpret_cast<B*>(p_result);
  }

  template<int A = AddressMode, int S = StorageMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeStatic, B*>::type
  array_data_ptr(size_t pos) const {
    assert(pos < ArraySize);
    assert(this->id() < Owner::storage().size());

    // Use constant-folded value for address computation.
    constexpr auto cptr_data_offset =
        StorageDataOffset<typename Owner::Storage>::value;
    constexpr auto cptr_storage_buffer = Owner::storage_buffer();
    constexpr auto array_location =
        cptr_storage_buffer + cptr_data_offset + (Capacity+1)*Offset;
    B* soa_array = reinterpret_cast<B*>(array_location
                                        + pos*sizeof(B)*(Capacity+1));

    return soa_array + reinterpret_cast<uintptr_t>(this);
  }

  template<int A = AddressMode, int S = StorageMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeDynamic, B*>::type
  array_data_ptr(size_t pos) const {
    assert(pos < ArraySize);
    assert(this->id() < Owner::storage().size());

    // Cannot constant fold dynamically allocated storage.
    auto p_base = Owner::storage().data_reference();
    return reinterpret_cast<B*>(
        p_base + (Capacity+1)*(Offset + pos*sizeof(B)) +
        reinterpret_cast<uintptr_t>(this)*sizeof(B));
  }

#include "soa/field_shared.inc"
#include "soa/array_shared.inc"
};

}  // namespace soa
}  // namespace ikra

#endif  // SOA_ARRAY_H
