#ifndef SOA_LAYOUT_H
#define SOA_LAYOUT_H

#include "soa/array_field.h"
#include "soa/constants.h"
#include "soa/field.h"
#include "soa/storage.h"

namespace ikra {
namespace soa {

// This marco is expanded for every primitive data type (such as int, float)
// and generates alias types for SOA field declarations. For example:
// int_<Offset> --> Field<int, Offset>
#define IKRA_DEFINE_LAYOUT_FIELD_TYPE(type) \
  template<int Offset> \
  using type ## _ = Field<type, Offset>; \

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
// polymorphism or "Curiously Recurring Template Pattern"). ContainerSize is
// the maximum number of instances this class can have. It is a compile-time
// constant to allow for efficient field address computation. AddressMode can
// be either "Zero Addressing Mode" or "Valid Addressing Mode".
template<class Self,
         uintptr_t ContainerSize,
         int AddressMode = kAddressModeZero,
         template<class Owner> class StorageStrategy = DynamicStorage>
class SoaLayout : SizeNDummy<AddressMode> {
 public:
  using Storage = StorageStrategy<Self>;
  const static uintptr_t kContainerSize = ContainerSize;

  // Define a Field_ alias as a shortcut.
  template<typename T, int Offset>
  using Field = Field_<T, ContainerSize, Offset, AddressMode, Self>;

  // Generate field types. Implement more types as necessary.
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(bool);
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(char);
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(double);
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(float);
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(int);

  // This struct serves as a namespace and contains array field types.
  struct array {
    template<typename T, size_t N, int Offset>
    using aos = ikra::soa::AosArrayField_<std::array<T, N>, ContainerSize,
                                          Offset, AddressMode, Self>;

    template<typename T, size_t N, int Offset>
    using soa = ikra::soa::SoaArrayField_<T, N, ContainerSize,
                                          Offset, AddressMode, Self>;
  };

  static const int kAddressMode = AddressMode;

  // Create a new instance of this class. Data will be allocated inside
  // storage.data.
  void* operator new(size_t count) {
    check_sizeof_class();
    assert(count == sizeof(Self));
    // Check if out of memory.
    assert(Self::storage.size <= ContainerSize);

    return get(Self::storage.size++);
  }

  // Create multiple new instances of this class. Data will be allocated inside
  // storage.data.
  void* operator new[](size_t count) {
    check_sizeof_class();
    // Size of this class is 1. "count" is the number of new instances.
    Self* first_ptr = get(Self::storage.size);
    Self::storage.size += count/AddressMode;
    return first_ptr;
  }

  // TODO: Implement delete operator.
  void operator delete(void* ptr) {
    assert(false);
  }

  // Return the number of instances of this class.
  static uintptr_t size() {
    return Self::storage.size;
  }

  // Return a pointer to an object with a given ID.
  static Self* get(uintptr_t id) {
    assert(id <= Self::storage.size);
    return get_(id);
  }

  // Return an iterator pointing to the first instance of this class.
  static executor::Iterator<Self*> begin() {
    return executor::Iterator<Self*>(Self::get(0));
  }

  // Return an iterator pointing to the last instance of this class + 1.
  static executor::Iterator<Self*> end() {
    return ++executor::Iterator<Self*>(Self::get(size() - 1));
  }

  // Calculate the ID of this object (assuming valid addressing mode).
  template<int A = AddressMode>
  typename std::enable_if<A != kAddressModeZero, uintptr_t>::type 
  id() const {
    return (reinterpret_cast<uintptr_t>(this) - 
           reinterpret_cast<uintptr_t>(Self::storage.data)) / AddressMode;
  }

  // Calculate the ID of this object (assuming zero addressing mode).
  template<int A = AddressMode>
  typename std::enable_if<A == kAddressModeZero, uintptr_t>::type 
  id() const {
    return reinterpret_cast<uintptr_t>(this);
  }

 private:
  // Return a pointer to an object by ID (assuming valid addressing mode).
  template<int A = AddressMode>
  static typename std::enable_if<A != kAddressModeZero, Self*>::type
  get_(uintptr_t id) {
    uintptr_t address = reinterpret_cast<uintptr_t>(Self::storage.data) +
                        id*AddressMode;
    return reinterpret_cast<Self*>(address);
  }

  // Return a pointer to an object by ID (assuming zero addressing mode).
  template<int A = AddressMode>
  static typename std::enable_if<A == kAddressModeZero, Self*>::type
  get_(uintptr_t id) {
    return reinterpret_cast<Self*>(id);
  }

  // Compile-time check for the size of this class. This check should fail
  // if this class contains fields that are not declared with the SOA DSL.
  // Assuming valid addressing mode.
  template<int A = AddressMode>
  static typename std::enable_if<A != kAddressModeZero, void>::type
  check_sizeof_class() {
    static_assert(sizeof(Self) == AddressMode,
                  "SOA class must have only SOA fields.");
  }

  // Compile-time check for the size of this class. This check should fail
  // if this class contains fields that are not declared with the SOA DSL.
  // Assuming zero addressing mode.
  template<int A = AddressMode>
  static typename std::enable_if<A == kAddressModeZero, void>::type
  check_sizeof_class() {
    static_assert(sizeof(Self) == 0,
                  "SOA class must have only SOA fields.");
  }
};

#undef IKRA_DEFINE_LAYOUT_FIELD_TYPE

}  // namespace soa
}  // namespace ikra

#endif  // SOA_LAYOUT_H
