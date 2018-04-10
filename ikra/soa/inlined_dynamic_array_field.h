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

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, void>::type
  set_external_pointer(B* ptr) {
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
  set_external_pointer(B* ptr) {
    assert(reinterpret_cast<uintptr_t>(ptr) % 4 == 0);
    auto p_external = reinterpret_cast<uintptr_t>(this)*sizeof(B*) +
                      reinterpret_cast<uintptr_t>(Owner::storage().data_ptr())
                      + (Capacity+1)*Offset;
    assert(p_external % sizeof(B**) == 0);
    *reinterpret_cast<B**>(p_external) = ptr;
  }

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, B*>::type
  get_external_pointer() const {
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
  __ikra_device__ typename std::enable_if<A == kAddressModeZero, B*>::type
  get_external_pointer() const {
    assert(this->id() < Owner::storage().size());

    auto p_external = reinterpret_cast<uintptr_t>(this)*sizeof(B*) +
                      reinterpret_cast<uintptr_t>(Owner::storage().data_ptr())
                      + (Capacity+1)*Offset;
    return *reinterpret_cast<B**>(p_external);
  }

  // TODO: Implement iterator and other methods.

 protected:
  __ikra_device__ SoaInlinedDynamicArrayField_() {
    this->set_external_pointer(nullptr);
  }

  // Calculate the address of an array element. For details, see comment
  // of data_ptr in Field_.
  template<size_t Pos, int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, B*>::type
  array_data_ptr() const {
    // Ensure that this is a valid pointer: Only those objects may be accessed
    // which were created with the "new" keyword and are thus initialized.
    assert(this->id() < Owner::storage().size());

    /*
    auto p_this = reinterpret_cast<uintptr_t>(this);
    auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());

    if (Pos < InlinedSize) {
      // Within inlined storage.
      uintptr_t p_result = (p_this - p_base - A)/A*sizeof(B) + p_base +
                           (Capacity+1)*(Offset + Pos*sizeof(B));
      return reinterpret_cast<B*>(p_result);
    } else {
      // Within external storage. Pointer at position InlinedSize + 1.
      B* p_external = this->get_external_pointer();
      assert(p_external != nullptr);
      return p_external + (Pos - InlinedSize);
    }
    */
    // TODO: Implement
    assert(false);
  }

  template<size_t Pos, int A = AddressMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero, B*>::type
  array_data_ptr() const {
    assert(this->id() < Owner::storage().size());

    auto p_this = reinterpret_cast<uintptr_t>(this);
    auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());

    if (Pos < InlinedSize) {
      // Within inlined storage.
      return reinterpret_cast<B*>(p_this*sizeof(B) + p_base +
                                  (Capacity+1)*(Offset + Pos*sizeof(B)
                                                       + sizeof(B*)));
    } else {
      // Within external storage.
      B* p_external = this->get_external_pointer();
      assert(p_external != nullptr);
      return p_external + (Pos - InlinedSize);
    }
  }

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, B*>::type
  array_data_ptr(size_t pos) const {
    // Ensure that this is a valid pointer: Only those objects may be accessed
    // which were created with the "new" keyword and are thus initialized.
    assert(this->id() < Owner::storage().size());

    /*
    auto p_this = reinterpret_cast<uintptr_t>(this);
    auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());

    if (pos < InlinedSize) {
      // Within inlined storage.
      uintptr_t p_result = (p_this - p_base - A)/A*sizeof(B) + p_base +
                           (Capacity+1)*(Offset + pos*sizeof(B));
      return reinterpret_cast<B*>(p_result);
    } else {
      // Within external storage. Pointer at position InlinedSize + 1.
      B* p_external = this->get_external_pointer();
      assert(p_external != nullptr);
      return p_external + (pos - InlinedSize);
    }
    */

    // TODO: Implement
    assert(false);
  }

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero, B*>::type
  array_data_ptr(size_t pos) const {
    assert(this->id() < Owner::storage().size());

    auto p_this = reinterpret_cast<uintptr_t>(this);
    auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());

    if (pos < InlinedSize) {
      // Within inlined storage.
      return reinterpret_cast<B*>(p_this*sizeof(B) + p_base +
                                  (Capacity+1)*(Offset + pos*sizeof(B)
                                                       + sizeof(B*)));
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
