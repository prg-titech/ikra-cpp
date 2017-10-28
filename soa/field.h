#ifndef SOA_FIELD_H
#define SOA_FIELD_H

#include "soa/constants.h"

// This marco is expanded for every inplace assignment operator and forwards
// the operation to the wrapped data.
#define IKRA_DEFINE_FIELD_ASSIGNMENT(symbol) \
  Self operator symbol ## = (T value) { \
    *data_ptr() symbol ## = value; \
    return *this; \
  }


namespace ikra {
namespace soa {

template<typename T,
         IndexType Capacity,
         uint32_t Offset,
         int AddressMode,
         class Owner>
class Field_ {
 private:
  using Self = Field_<T, Capacity, Offset, AddressMode, Owner>;

 public:
  // This class may only be used to declare fields of classes.
  void* operator new(size_t count) = delete;

  // TODO: What should we do with the dereference operator?
  //T& operator*() const {
  //  return *data_ptr();
  //}

  T* operator->() const {
    return data_ptr();
  }

  T* operator&() const {
    return data_ptr();
  }

  // Get the value of this field. This method is not usually needed and the
  // preferred way to retrieve a value is through implicit conversion.
  T& get() const {
    return *data_ptr();
  }

  // Operator for implicit conversion to type T.
  operator T&() const {
    return *data_ptr();
  }

  Self operator=(T value) {
    *data_ptr() = value;
    return *this;
  }

  // Define inplace assignment operators.
  IKRA_DEFINE_FIELD_ASSIGNMENT(+);
  IKRA_DEFINE_FIELD_ASSIGNMENT(-);
  IKRA_DEFINE_FIELD_ASSIGNMENT(*);
  IKRA_DEFINE_FIELD_ASSIGNMENT(/);
  IKRA_DEFINE_FIELD_ASSIGNMENT(%);
  IKRA_DEFINE_FIELD_ASSIGNMENT(&);
  IKRA_DEFINE_FIELD_ASSIGNMENT(|);
  IKRA_DEFINE_FIELD_ASSIGNMENT(^);
  IKRA_DEFINE_FIELD_ASSIGNMENT(<<);
  IKRA_DEFINE_FIELD_ASSIGNMENT(>>);

  // TODO: Implement special operators:
  // http://en.cppreference.com/w/cpp/language/operator_logical

 protected:
  // Only Owner can create new fields for itself.
  friend Owner;
  Field_() {}

  // Initialize the field with a given value. Used in class constructor.
  Field_(T value) {
    *this = value;
  }

  // Initialize the field with the value of another field. Used in class
  // constructor. TODO: Unclear why exactly we need this one...
  Field_(const Field_& other) {}

  // Not sure why we need this. Required to make field inplace initialization
  // work.
  Field_(Field_&& other) {}

  template<int A = AddressMode>
  typename std::enable_if<A != kAddressModeZero, IndexType>::type
  id() const {
    return (reinterpret_cast<uintptr_t>(this) - 
           reinterpret_cast<uintptr_t>(Owner::storage.data)) / A - 1;
  }

  template<int A = AddressMode>
  typename std::enable_if<A == kAddressModeZero, IndexType>::type
  id() const {
    return reinterpret_cast<uintptr_t>(this);
  }

  // Calculate the address of this field based on the "this" pointer of this
  // Field instance.
  template<int A = AddressMode>
  typename std::enable_if<A != kAddressModeZero, T*>::type
  data_ptr() const {
    // Ensure that this is a valid pointer: Only those objects may be accessed
    // which were created with the "new" keyword and are thus initialized.
    assert(id() < Owner::storage.size);

    uintptr_t p_this = reinterpret_cast<uintptr_t>(this);
    uintptr_t p_base = reinterpret_cast<uintptr_t>(Owner::storage.data);
    uintptr_t p_result = (p_this - p_base - A)/A*sizeof(T) + p_base +
                         Capacity*Offset;
    return reinterpret_cast<T*>(p_result);
  }

  template<int A = AddressMode>
  typename std::enable_if<A == kAddressModeZero, T*>::type
  data_ptr() const {
    assert(id() < Owner::storage.size);
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(this)*sizeof(T) +
                                Owner::storage.data + Capacity*Offset);
  }

  // Force size of this class to be 0.
  char dummy_[0];
};

}  // namespace soa
}  // namespace ikra

#undef IKRA_DEFINE_FIELD_ASSIGNMENT

#endif  // SOA_FIELD_H
