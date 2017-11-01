#ifndef SOA_INLINED_DYNAMIC_ARRAY_FIELD_H
#define SOA_INLINED_DYNAMIC_ARRAY_FIELD_H

#include "soa/constants.h"
#include "soa/field.h"

namespace ikra {
namespace soa {

// Class for field declarations of type "Inlined Dynamic Array". This type
// stores InlinedSize many items in SOA format (like SoaArrayField_) and
// the remaining items in a user-specified location. This array does not know
// its size.
// Description of template parameters:
// * T: Base type of the array.
// * InlinedSize: Number of elements that should be stored in SOA format.
// * Remaining fields: See array_field.h
// Note, for SOA class references, T must be a pointer type!
template<typename T,
         size_t InlinedSize,
         IndexType Capacity,
         uint32_t Offset,
         int AddressMode,
         class Owner,
         class ArraySizeT = IndexType>
class SoaInlinedDynamicArrayField_ : public Field_<T, Capacity, Offset,
                                                   AddressMode, Owner> {
 private:
  using Self = SoaInlinedDynamicArrayField_<T, InlinedSize, Capacity, Offset,
                                            AddressMode, Owner>;

 public:
  static const int kSize = InlinedSize*sizeof(T) + sizeof(T**);

  SoaInlinedDynamicArrayField_(T* external_storage) {
    this->set_external_pointer(external_storage);
  }

  SoaInlinedDynamicArrayField_(size_t num_elements) {
    // Allocate memory in arena if necessary.
    if (num_elements > InlinedSize) {
      void* mem = Owner::storage().allocate_in_arena(
          (num_elements - InlinedSize)*sizeof(T));
      this->set_external_pointer(reinterpret_cast<T*>(mem));
    }
  }

  // Support calling methods using -> syntax.
  const Self* operator->() const {
    return this;
  }

  T* operator&() const = delete;

  T& get() const = delete;

  operator T&() const = delete;

  // Implement std::array interface.

  T& operator[](size_t pos) const {
    return *this->array_data_ptr(pos);
  }

  T& at(size_t pos) const {
    // TODO: Not sure about the semantics of this function. Instances do not
    // know their size.
    return this->operator[](pos);
  }

  template<size_t Pos>
  T& at() const {
    return *array_data_ptr<Pos>();
  }

  T& front() const {
    return at<0>();
  }

  template<int A = AddressMode>
  typename std::enable_if<A != kAddressModeZero, void>::type
  set_external_pointer(T* ptr) {
    uintptr_t p_this = reinterpret_cast<uintptr_t>(this);
    uintptr_t p_base = reinterpret_cast<uintptr_t>(Owner::storage().data);
    uintptr_t p_external = (p_this - p_base - A)/A*sizeof(T) + p_base +
                           Capacity*(Offset + InlinedSize*sizeof(T));
    *reinterpret_cast<T**>(p_external) = ptr;
  }

  template<int A = AddressMode>
  typename std::enable_if<A == kAddressModeZero, void>::type
  set_external_pointer(T* ptr) {
    uintptr_t p_external = reinterpret_cast<uintptr_t>(this)*sizeof(T) +
                           reinterpret_cast<uintptr_t>(Owner::storage().data) +
                           Capacity*(Offset + InlinedSize*sizeof(T));
    *reinterpret_cast<T**>(p_external) = ptr;
  }

  template<int A = AddressMode>
  typename std::enable_if<A != kAddressModeZero, T*>::type
  get_external_pointer() const {
    assert(this->id() < Owner::storage().size);

    uintptr_t p_this = reinterpret_cast<uintptr_t>(this);
    uintptr_t p_base = reinterpret_cast<uintptr_t>(Owner::storage().data);
    uintptr_t p_external = (p_this - p_base - A)/A*sizeof(T) + p_base +
                           Capacity*(Offset + InlinedSize*sizeof(T));
    return *reinterpret_cast<T**>(p_external);
  }

  template<int A = AddressMode>
  typename std::enable_if<A == kAddressModeZero, T*>::type
  get_external_pointer() const {
    assert(this->id() < Owner::storage().size);

    uintptr_t p_external = reinterpret_cast<uintptr_t>(this)*sizeof(T) +
                           reinterpret_cast<uintptr_t>(Owner::storage().data) +
                           Capacity*(Offset + InlinedSize*sizeof(T));
    return *reinterpret_cast<T**>(p_external);
  }

  // TODO: Implement iterator and other methods.

 protected:
  SoaInlinedDynamicArrayField_() {
    this->set_external_pointer(nullptr);
  }

  // Calculate the address of an array element. For details, see comment
  // of data_ptr in Field_.
  template<size_t Pos, int A = AddressMode>
  typename std::enable_if<A != kAddressModeZero, T*>::type
  array_data_ptr() const {
    // Ensure that this is a valid pointer: Only those objects may be accessed
    // which were created with the "new" keyword and are thus initialized.
    assert(this->id() < Owner::storage().size);

    uintptr_t p_this = reinterpret_cast<uintptr_t>(this);
    uintptr_t p_base = reinterpret_cast<uintptr_t>(Owner::storage().data);

    if (Pos < InlinedSize) {
      // Within inlined storage.
      uintptr_t p_result = (p_this - p_base - A)/A*sizeof(T) + p_base +
                           Capacity*(Offset + Pos*sizeof(T));
      return reinterpret_cast<T*>(p_result);
    } else {
      // Within external storage. Pointer at position InlinedSize + 1.
      T* p_external = this->get_external_pointer();
      assert(p_external != nullptr);
      return p_external + (Pos - InlinedSize);
    }
  }

  template<size_t Pos, int A = AddressMode>
  typename std::enable_if<A == kAddressModeZero, T*>::type
  array_data_ptr() const {
    assert(this->id() < Owner::storage().size);

    uintptr_t p_this = reinterpret_cast<uintptr_t>(this);
    uintptr_t p_base = reinterpret_cast<uintptr_t>(Owner::storage().data);

    if (Pos < InlinedSize) {
      // Within inlined storage.
      return reinterpret_cast<T*>(p_this*sizeof(T) + p_base +
                                  Capacity*(Offset + Pos*sizeof(T)));
    } else {
      // Within external storage.
      T* p_external = this->get_external_pointer();
      assert(p_external != nullptr);
      return p_external + (Pos - InlinedSize);
    }
  }

  template<int A = AddressMode>
  typename std::enable_if<A != kAddressModeZero, T*>::type
  array_data_ptr(size_t pos) const {
    // Ensure that this is a valid pointer: Only those objects may be accessed
    // which were created with the "new" keyword and are thus initialized.
    assert(this->id() < Owner::storage().size);

    uintptr_t p_this = reinterpret_cast<uintptr_t>(this);
    uintptr_t p_base = reinterpret_cast<uintptr_t>(Owner::storage().data);

    if (pos < InlinedSize) {
      // Within inlined storage.
      uintptr_t p_result = (p_this - p_base - A)/A*sizeof(T) + p_base +
                           Capacity*(Offset + pos*sizeof(T));
      return reinterpret_cast<T*>(p_result);
    } else {
      // Within external storage. Pointer at position InlinedSize + 1.
      T* p_external = this->get_external_pointer();
      assert(p_external != nullptr);
      return p_external + (pos - InlinedSize);
    }
  }

  template<int A = AddressMode>
  typename std::enable_if<A == kAddressModeZero, T*>::type
  array_data_ptr(size_t pos) const {
    assert(this->id() < Owner::storage().size);

    uintptr_t p_this = reinterpret_cast<uintptr_t>(this);
    uintptr_t p_base = reinterpret_cast<uintptr_t>(Owner::storage().data);

    if (pos < InlinedSize) {
      // Within inlined storage.
      return reinterpret_cast<T*>(p_this*sizeof(T) + p_base +
                                  Capacity*(Offset + pos*sizeof(T)));
    } else {
      // Within external storage.
      T* p_external = this->get_external_pointer();
      assert(p_external != nullptr);
      return p_external + (pos - InlinedSize);
    }
  }
};

}  // namespace soa
}  // namespace ikra

#endif  // SOA_INLINED_DYNAMIC_ARRAY_FIELD_H
