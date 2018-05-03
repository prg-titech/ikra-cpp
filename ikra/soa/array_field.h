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
         int LayoutMode,
         class Owner>
class ArrayObjectField_ {
 private:
  using Self = ArrayObjectField_<T, B, ArraySize, Capacity, Offset,
                                 AddressMode, StorageMode, LayoutMode, Owner>;
 public:
  static const uint32_t DBG_OFFSET = Offset;  // For debugging.

  static const int kSize = sizeof(T);
  static_assert(kSize == sizeof(B)*ArraySize,
                "Internal error: Array size mismatch.");

  __ikra_device__ IndexType size() const { return ArraySize; }

 protected:
  template<size_t Pos>
  __ikra_device__ B* array_data_ptr() const {
    static_assert(Pos < ArraySize, "Array index out of bounds.");
    return reinterpret_cast<B*>(data_ptr()) + Pos;
  }

  __ikra_device__ B* array_data_ptr(size_t pos) const {
    assert(pos < ArraySize);
    return reinterpret_cast<B*>(data_ptr()) + pos;
  }

#include "soa/addressable_field_shared.inc"
#include "soa/array_shared.inc"
#include "soa/field_shared.inc"
};

// Class for field declarations of type array. B is the base type of the array.
// This class is the SoA counter part of ArrayObjectField_. Array slots are
// layouted as if they were SoA fields (columns).
template<typename B,
         size_t ArraySize,
         IndexType Capacity,
         uint32_t Offset,
         int AddressMode,
         int StorageMode,
         int LayoutMode,
         class Owner>
class FullyInlinedArrayField_ {
 private:
  using Self = FullyInlinedArrayField_<B, ArraySize, Capacity, Offset,
                                       AddressMode, StorageMode,
                                       LayoutMode, Owner>;

  // TODO: Implement host-side functions.
  template<int OuterVirtualWarpSize, int VirtualWarpSize>
  class VirtualWarpRangeArray {
   private:
    // TODO: Consider using pointers instead of array reference + index as
    // internal state.
    class Iterator {
     public:
      __ikra_device__ Iterator(Self& array_self, IndexType index)
          : array_self_(array_self), index_(index) {}

      __ikra_device__ Iterator& operator++() {    // Prefix increment.
        index_ += OuterVirtualWarpSize/VirtualWarpSize;
        return *this;
      }

      __ikra_device__ B& operator*() const {
        return array_self_[index_];
      }

      __ikra_device__ bool operator==(const Iterator& other) const {
        // TODO: Should also check if array is the same one.
        return index_ == other.index_;
      }

      __ikra_device__ bool operator!=(const Iterator& other) const {
        return !(*this == other);
      }

     private:
      Self& array_self_;
      IndexType index_;
    };

    Self& array_self_;

   public:
    __ikra_device__ VirtualWarpRangeArray(Self& array_self)
        : array_self_(array_self) {
      assert(VirtualWarpSize <= OuterVirtualWarpSize);
    }

    __ikra_device__ Iterator begin() const {
#if __CUDA_ARCH__
      const int tid = threadIdx.x % OuterVirtualWarpSize;
#else
      const int tid = 0;
#endif  // __CUDA_ARCH__
      return Iterator(array_self_, tid/VirtualWarpSize);
    }

    __ikra_device__ Iterator end() const {
#if __CUDA_ARCH__
      const int first_idx =
          (threadIdx.x % OuterVirtualWarpSize)/VirtualWarpSize;
#else
      const int first_idx = 0;
#endif  // __CUDA_ARCH__
      // Formula: R(ArraySize - first_idx) + first_idx
      //          R(i): Round up to next multiple of OuterVWS/VWS
      const int step_size = OuterVirtualWarpSize/VirtualWarpSize;
      const IndexType end_idx = ((ArraySize - first_idx + step_size - 1)
          /step_size)*step_size + first_idx;
      return Iterator(array_self_, end_idx);
    }
  };

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

  __ikra_device__ IndexType size() const { return ArraySize; }

  // TODO: Implement iterator and other methods..
#ifdef __CUDA_ARCH__
  // TODO: Make this function and self references const.
  template<int OuterVirtualWarpSize, int VirtualWarpSize = 1>
  __ikra_device__ VirtualWarpRangeArray<OuterVirtualWarpSize, VirtualWarpSize>
      vw_iterator() {
    return VirtualWarpRangeArray<OuterVirtualWarpSize, VirtualWarpSize>(*this);
  }
#else
  template<int OuterVirtualWarpSize, int VirtualWarpSize = 1>
  __ikra_device__ VirtualWarpRangeArray<OuterVirtualWarpSize, VirtualWarpSize>
      vw_iterator() {
    assert(OuterVirtualWarpSize == 0 && VirtualWarpSize == 1);
    return VirtualWarpRangeArray<1, 1>(*this);
  }
#endif

 protected:
  // Calculate the address of an array element. For details, see comment
  // of data_ptr in Field_.
  template<size_t Pos, int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, B*>::type
  array_data_ptr() const {
    static_assert(Pos < ArraySize, "Array index out of bounds.");
    auto p_this = reinterpret_cast<uintptr_t>(this);
    auto p_base = Owner::storage().data_reference();
    auto p_result = (p_base + p_this - p_base - A)/A*sizeof(B) +
                    (Capacity+1)*(Offset + Pos*sizeof(B));
    return reinterpret_cast<B*>(p_result);
  }

  template<size_t Pos, int A = AddressMode, int S = StorageMode,
           int L = LayoutMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeStatic &&
                                          L == kLayoutModeSoa, B*>::type
  array_data_ptr() const {
    static_assert(Pos < ArraySize, "Array index out of bounds.");

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

  template<size_t Pos, int A = AddressMode, int S = StorageMode,
           int L = LayoutMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeStatic &&
                                          L == kLayoutModeAos, B*>::type
  array_data_ptr() const {
    static_assert(Pos < ArraySize, "Array index out of bounds.");

    B* inner_array = reinterpret_cast<B*>(
        reinterpret_cast<char*>(const_cast<Self*>(this)) + Offset);
    return inner_array + Pos;
  }

  template<size_t Pos, int A = AddressMode, int S = StorageMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeDynamic, B*>::type
  array_data_ptr() const {
    static_assert(Pos < ArraySize, "Array index out of bounds.");

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

    auto p_this = reinterpret_cast<uintptr_t>(this);
    auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());
    auto p_base_ref = Owner::storage().data_reference();
    auto p_result = p_base_ref + (p_this - p_base - A)/A*sizeof(B) +
                    (Capacity+1)*(Offset + pos*sizeof(B));
    return reinterpret_cast<B*>(p_result);
  }

  template<int A = AddressMode, int S = StorageMode, int L = LayoutMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeStatic &&
                                          L == kLayoutModeSoa, B*>::type
  array_data_ptr(size_t pos) const {
    assert(pos < ArraySize);

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

  template<int A = AddressMode, int S = StorageMode, int L = LayoutMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeStatic &&
                                          L == kLayoutModeAos, B*>::type
  array_data_ptr(size_t pos) const {
    assert(pos < ArraySize);

    B* inner_array = reinterpret_cast<B*>(
        reinterpret_cast<char*>(const_cast<Self*>(this)) + Offset);
    return inner_array + pos;
  }

  template<int A = AddressMode, int S = StorageMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeDynamic, B*>::type
  array_data_ptr(size_t pos) const {
    assert(pos < ArraySize);

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
