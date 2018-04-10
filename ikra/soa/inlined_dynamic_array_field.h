#ifndef SOA_INLINED_DYNAMIC_ARRAY_FIELD_H
#define SOA_INLINED_DYNAMIC_ARRAY_FIELD_H

#include <limits>

#include "soa/constants.h"
#include "soa/field.h"

namespace ikra {
namespace soa {

// Class for field declarations of type "Inlined Dynamic Array". This type
// stores InlinedSize many items in SOA format (like SoaArrayField_) and
// the remaining items in a user-specified location. This array does not know
// its size.
// Description of template parameters:
// * B: Base type of the array.
// * InlinedSize: Number of elements that should be stored in SOA format.
// * Remaining fields: See array_field.h
// Note, for SOA class references, B must be a pointer type!
template<typename B,
         size_t InlinedSize,
         IndexType Capacity,
         uint32_t Offset,
         int AddressMode,
         int StorageMode,
         class Owner,
         class ArraySizeT = IndexType>
class SoaInlinedDynamicArrayField_ {
 private:
  using Self = SoaInlinedDynamicArrayField_<B, InlinedSize, Capacity, Offset,
                                            AddressMode, StorageMode, Owner>;

  // TODO: Move functions that require this constant in a separate file and
  // do not include here.
  static const size_t ArraySize = std::numeric_limits<size_t>::max();

 public:
  static const int kSize = InlinedSize*sizeof(B) + sizeof(B*);

  __ikra_device__ SoaInlinedDynamicArrayField_(B* external_storage) {
    this->set_external_pointer(external_storage);
  }

  __ikra_device__ SoaInlinedDynamicArrayField_(size_t num_elements) {
    // Allocate memory in arena if necessary.
    if (num_elements > InlinedSize) {
      void* mem = Owner::storage().allocate_in_arena(
          (num_elements - InlinedSize)*sizeof(B));
      this->set_external_pointer(reinterpret_cast<B*>(mem));
    }
  }

  // Support calling methods using -> syntax.
  __ikra_device__ const Self* operator->() const {
    return this;
  }

  B* operator&() const = delete;

  B& get() const = delete;

  operator B&() const = delete;

  // TODO: Implement iterator and other methods.

 protected:
  __ikra_device__ SoaInlinedDynamicArrayField_() {
    this->set_external_pointer(nullptr);
  }

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, void>::type
  set_external_pointer_internal(B* ptr) {
    /*
    assert(reinterpret_cast<uintptr_t>(ptr) % 4 == 0);
    auto p_this = reinterpret_cast<uintptr_t>(this);
    auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());
    auto p_external = (p_this - p_base - A)/A*sizeof(B) + p_base +
                       (Capacity+1)*(Offset + InlinedSize*sizeof(B));
    assert(p_external % sizeof(B**) == 0);
    *reinterpret_cast<B**>(p_external) = ptr;
    */
    // TODO: Implement
    assert(false);
  }

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero, void>::type
  set_external_pointer_internal(B* ptr) {
    assert(reinterpret_cast<uintptr_t>(ptr) % 4 == 0);
    auto p_external = reinterpret_cast<uintptr_t>(this)*sizeof(B*) +
                      reinterpret_cast<uintptr_t>(Owner::storage().data_ptr())
                      + (Capacity+1)*(Offset + InlinedSize*sizeof(B));
    assert(p_external % sizeof(B**) == 0);
    *reinterpret_cast<B**>(p_external) = ptr;
  }

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, B**>::type
  get_external_pointer_addr_internal() const {
    assert(this->id() < Owner::storage().size());

    /*
    auto p_this = reinterpret_cast<uintptr_t>(this);
    auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());
    auto p_external = (p_this - p_base - A)/A*sizeof(B) + p_base +
                      (Capacity+1)*(Offset + InlinedSize*sizeof(B));
    return *reinterpret_cast<B**>(p_external);
    */
    // TODO: Implement
    assert(false);
    return nullptr;
  }

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero, B**>::type
  get_external_pointer_addr_internal() const {
    assert(this->id() < Owner::storage().size());

    auto p_external = reinterpret_cast<uintptr_t>(this)*sizeof(B*) +
                      reinterpret_cast<uintptr_t>(Owner::storage().data_ptr())
                      + (Capacity+1)*(Offset + InlinedSize*sizeof(B));
    assert(p_external % sizeof(B**) == 0);
    return reinterpret_cast<B**>(p_external);
  }

#if defined(__CUDA_ARCH__) || !defined(__CUDACC__)
// Not running in CUDA mode or running on device.
  __ikra_device__ B* get_external_pointer() const {
    return *get_external_pointer_addr_internal();
  }

  __ikra_device__ void set_external_pointer(B* ptr) {
    set_external_pointer_internal(ptr);
  }
#else
// Running on host, but data is located on GPU.
  // TODO: This seems really inefficient. Use only for debugging at the moment!
  template<int A = AddressMode>
  typename std::enable_if<A == kAddressModeZero, B*>::type
  get_external_pointer() const {
    // Copy external pointer from GPU.
    B** dev_p_external = Owner::storage().translate_address_host_to_device(
        reinterpret_cast<B**>(get_external_pointer_addr_internal()));
    B* dev_result = nullptr;
    cudaMemcpy(&dev_result, dev_p_external, sizeof(B*),
               cudaMemcpyDeviceToHost);
    assert(cudaPeekAtLastError() == cudaSuccess);

    // Now convert the value back to host-relative address.
    return Owner::storage().translate_address_device_to_host(dev_result);
  }

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero, void>::type
  set_external_pointer(B* ptr) {
    // TODO: Not implemented.
    assert(false);
  }
#endif

  // Calculate the address of an array element. For details, see comment
  // of data_ptr in Field_.
  template<size_t Pos, int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, B*>::type
  array_data_ptr() const {
    static_assert(Pos < ArraySize, "Array index out of bounds.");
    // Ensure that this is a valid pointer: Only those objects may be accessed
    // which were created with the "new" keyword and are thus initialized.
    assert(this->id() < Owner::storage().size());

    if (Pos < InlinedSize) {
      // Within inlined storage.
      auto p_this = reinterpret_cast<uintptr_t>(this);
      auto p_base = Owner::storage().data_reference();
      auto p_result = (p_base + p_this - p_base - A)/A*sizeof(B) +
                      (Capacity+1)*(Offset + Pos*sizeof(B));
      return reinterpret_cast<B*>(p_result);
    } else {
      // Within external storage. Pointer at position InlinedSize + 1.
      B* p_external = this->get_external_pointer();
      assert(p_external != nullptr);
      return p_external + (Pos - InlinedSize);
    }
  }

  template<size_t Pos, int A = AddressMode, int S = StorageMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeStatic, B*>::type
  array_data_ptr() const {
    static_assert(Pos < ArraySize, "Array index out of bounds.");
    assert(this->id() < Owner::storage().size());

    if (Pos < InlinedSize) {
      // Within inlined storage.
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
    } else {
      // Within external storage.
      B* p_external = this->get_external_pointer();
      assert(p_external != nullptr);
      return p_external + (Pos - InlinedSize);
    }
  }

  template<size_t Pos, int A = AddressMode, int S = StorageMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeDynamic, B*>::type
  array_data_ptr() const {
    static_assert(Pos < ArraySize, "Array index out of bounds.");
    assert(this->id() < Owner::storage().size());

    if (Pos < InlinedSize) {
      // Cannot constant fold dynamically allocated storage.
      auto p_base = Owner::storage().data_reference();
      return reinterpret_cast<B*>(
          p_base + (Capacity+1)*(Offset + Pos*sizeof(B)) +
          reinterpret_cast<uintptr_t>(this)*sizeof(B));
    } else {
      // Within external storage. Pointer at position InlinedSize + 1.
      B* p_external = this->get_external_pointer();
      assert(p_external != nullptr);
      return p_external + (Pos - InlinedSize);
    }
  }

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, B*>::type
  array_data_ptr(size_t pos) const {
    assert(pos < ArraySize);
    // Ensure that this is a valid pointer: Only those objects may be accessed
    // which were created with the "new" keyword and are thus initialized.
    assert(this->id() < Owner::storage().size());

    if (pos < InlinedSize) {
      // Within inlined storage.
      auto p_this = reinterpret_cast<uintptr_t>(this);
      auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());
      auto p_base_ref = Owner::storage().data_reference();
      auto p_result = p_base_ref + (p_this - p_base - A)/A*sizeof(B) +
                      (Capacity+1)*(Offset + pos*sizeof(B));
      return reinterpret_cast<B*>(p_result);
    } else {
      // Within external storage. Pointer at position InlinedSize + 1.
      B* p_external = this->get_external_pointer();
      assert(p_external != nullptr);
      return p_external + (pos - InlinedSize);
    }
  }

  template<int A = AddressMode, int S = StorageMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeStatic, B*>::type
  array_data_ptr(size_t pos) const {
    assert(pos < ArraySize);
    assert(this->id() < Owner::storage().size());

    if (pos < InlinedSize) {
      // Within inlined storage.
      // Use constant-folded value for address computation.
      constexpr auto cptr_data_offset =
          StorageDataOffset<typename Owner::Storage>::value;
      constexpr auto cptr_storage_buffer = Owner::storage_buffer();
      constexpr auto array_location =
          cptr_storage_buffer + cptr_data_offset + (Capacity+1)*Offset;
      B* soa_array = reinterpret_cast<B*>(array_location
                                          + pos*sizeof(B)*(Capacity+1));

      return soa_array + reinterpret_cast<uintptr_t>(this);
    } else {
      // Within external storage.
      B* p_external = this->get_external_pointer();
      assert(p_external != nullptr);
      return p_external + (pos - InlinedSize);
    }
  }

  template<int A = AddressMode, int S = StorageMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeDynamic, B*>::type
  array_data_ptr(size_t pos) const {
    assert(pos < ArraySize);
    assert(this->id() < Owner::storage().size());

    if (pos < InlinedSize) {
      // Within inlined storage.
      // Cannot constant fold dynamically allocated storage.
      auto p_base = Owner::storage().data_reference();
      return reinterpret_cast<B*>(
          p_base + (Capacity+1)*(Offset + pos*sizeof(B)) +
          reinterpret_cast<uintptr_t>(this)*sizeof(B));
    } else {
      // Within external storage.
      B* p_external = this->get_external_pointer();
      assert(p_external != nullptr);
      return p_external + (pos - InlinedSize);
    }
  }

#include "soa/field_shared.inc"
#include "soa/array_shared.inc"
};

}  // namespace soa
}  // namespace ikra

#endif  // SOA_INLINED_DYNAMIC_ARRAY_FIELD_H
