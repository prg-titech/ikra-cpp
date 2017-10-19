
#ifndef SOA_SOA_H
#define SOA_SOA_H

#include <tuple>
#include <type_traits>

// Asserts active only in debug mode (NDEBUG).
#include <cassert>


namespace ikra {
namespace soa {

// This marco is expanded for every inplace assignment operator and forwards
// the operation to the wrapped data.
#define IKRA_DEFINE_FIELD_ASSIGNMENT(symbol) \
  Self operator symbol ## = (T value) { \
    *data_ptr() symbol ## = value; \
    return *this; \
  }

// This marco is expanded for every primitive data type (such as int, float)
// and generates alias types for SOA field declarations. For example:
// int__<Offset> --> Field<int, Offset>
#define IKRA_DEFINE_LAYOUT_FIELD_TYPE(type) \
  template<int Offset> \
  using type ## _ = Field<type, Offset>; \

// In Zero Addressing Mode, the address of an object is its ID. E.g., the
// address of the first object (ID 0) is nullptr. Pointer arithmetics is not
// possible in Zero Addressing Mode, because the size of a SOA class is zero.
// I.e., sizeof(MyClass) = 0. Increasing the size to 1 does not work, because
// instance create will attempt to zero-initialize data at an invalid address.
// Zero Addressing Mode results in efficient assembly code for field reads/
// writes. The address of a field `i` in object with ID t is defined as:
// this*sizeof(type(i)) + ContainerSize*Offset(i) + ClassStorageBase.
// Notice that the second operand is a compile-time constant if the container
// size (max. number of elements of a class) is a compile-time constant.
// See test/soa/benchmarks/codegen_test.cc for inspection of assembly code.
static const int kAddressModeZero = 0;

// In Valid Addressing Mode, the address of an object is its ID plus the
// beginning of the class storage data chunk (ClassStorageBase). The size of
// an object is now defined as 1, allowing for pointer arithmetics on objects.
// Generated assembly code is less efficient, because the address of a field is
// now complex:
// (this - ClassStorageBase)*sizeof(type(i)) + ContainerSize*Offset(i) +
//     ClassStorageBase
// This can be rewritten as:
// ClassStorageBase*(1-sizeof(type(i)) + this*sizeof(type(i)) +
//     ContainerSize*Offset(i)
// See test/soa/pointer_arithmetics_test.cc for pointer arithmetics examples.
static const int kAddressModeValid = 1;


namespace {

template<typename T,
         uintptr_t ContainerSize,
         uint32_t Offset,
         int AddressMode,
         class Owner>
class Field_ {
 private:
  using Self = Field_<T, ContainerSize, Offset, AddressMode, Owner>;

 public:
  Field_(Field_&& other) = delete;

  // This class may only be used to declare fields of classes.
  void* operator new(size_t count) = delete;

  // TODO: What should we do with the dereference operator?

  T operator->() const {
    static_assert(std::is_pointer<T>::value,
                  "Cannot dereference non-pointer type value.");
    return *data_ptr();
  }

  T* operator&() const {
    return data_ptr();
  }

  // Get the value of this field. This method is not usually needed and the
  // preferred way to retrieve a value is through implicit conversion.
  T get() const {
    return *data_ptr();
  }

  // Operator for implicit conversion to type T.
  operator T() const {
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

 private:
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

  template<int A = AddressMode>
  typename std::enable_if<A == kAddressModeValid, uintptr_t>::type 
  id() const {
    static_assert(AddressMode == kAddressModeValid, "Internal error.");
    return reinterpret_cast<uintptr_t>(this) - 
           reinterpret_cast<uintptr_t>(Owner::storage.data);
  }

  template<int A = AddressMode>
  typename std::enable_if<A == kAddressModeZero, uintptr_t>::type 
  id() const {
    static_assert(AddressMode == kAddressModeZero, "Internal error.");
    return reinterpret_cast<uintptr_t>(this);
  }

  // Calculate the address of this field based on the "this" pointer of this
  // Field instance.
  template<int A = AddressMode>
  typename std::enable_if<A == kAddressModeValid, T*>::type
  data_ptr() const {
    static_assert(AddressMode == kAddressModeValid, "Internal error.");
    // Ensure that this is a valid pointer: Only those objects may be accessed
    // which were created with the "new" keyword and are thus initialized.
    assert(id() < Owner::storage.size);

    uintptr_t column_skip = ContainerSize*Offset;
    uintptr_t p1 = reinterpret_cast<uintptr_t>(Owner::storage.data) *
                   (1 - sizeof(T));
    uintptr_t p2 = reinterpret_cast<uintptr_t>(this)*sizeof(T);
    return reinterpret_cast<T*>(p1 + p2  + column_skip);
  }

  template<int A = AddressMode>
  typename std::enable_if<A == kAddressModeZero, T*>::type
  data_ptr() const {
    static_assert(AddressMode == kAddressModeZero, "Internal error.");
    assert(id() < Owner::storage.size);
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(this)*sizeof(T) +
                                Owner::storage.data + ContainerSize*Offset);
  }

  // Force size of this class to be 0.
  char dummy_[0];
};

#undef IKRA_DEFINE_FIELD_ASSIGNMENT

// sizeof(Size0Dummy) = 0;
struct Size0Dummy {
  char dummy_[0];
};

// sizeof(Size1Dummy) = 1;
struct Size1Dummy {};

}  // namespace

template<class Self,
         uint32_t ObjectSize,
         uintptr_t ContainerSize,
         int AddressMode = kAddressModeValid>
class SoaLayout
    : std::conditional<AddressMode == kAddressModeZero,
                       Size0Dummy, Size1Dummy>::type {
 public:
  void* operator new(size_t count) {
    check_sizeof_class();
    assert(count == sizeof(Self));
    // Check if out of memory.
    assert(Self::storage.size <= ContainerSize);

    return get(Self::storage.size++);
  }

  void* operator new[](size_t count) {
    check_sizeof_class();
    // Size of this class is 1. "count" is the number of new instances.
    Self* first_ptr = get(Self::storage.size);
    Self::storage.size += count;
    return first_ptr;
  }

  // TODO: Implement delete operator.
  void operator delete(void* ptr) {
    assert(false);
  }

  template<uint32_t TotalSize>
  class InternalStorage {
   public:
    uint32_t size;
    char data[TotalSize];
  };

  using Storage = InternalStorage<ObjectSize*ContainerSize>;

  template<typename T, int Offset>
  using Field = Field_<T, ContainerSize, Offset, AddressMode, Self>;

  // Implement more types as necessary.
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(bool);
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(char);
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(double);
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(float);
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(int);

  // Return a pointer to an object with a given ID.
  static Self* get(uintptr_t id) {
    assert(id <= Self::storage.size);
    return get_(id);
  }

  template<int A = AddressMode>
  typename std::enable_if<A == kAddressModeValid, uintptr_t>::type 
  id() const {
    return reinterpret_cast<uintptr_t>(this) - 
           reinterpret_cast<uintptr_t>(Self::storage.data);
  }

  template<int A = AddressMode>
  typename std::enable_if<A == kAddressModeZero, uintptr_t>::type 
  id() const {
    return reinterpret_cast<uintptr_t>(this);
  }

 private:
  // Return a pointer to an object with a given ID.
  template<int A = AddressMode>
  static typename std::enable_if<A == kAddressModeValid, Self*>::type
  get_(uintptr_t id) {
    uintptr_t address = reinterpret_cast<uintptr_t>(Self::storage.data) + id;
    return reinterpret_cast<Self*>(address);
  }

  template<int A = AddressMode>
  static typename std::enable_if<A == kAddressModeZero, Self*>::type
  get_(uintptr_t id) {
    return reinterpret_cast<Self*>(id);
  }

  template<int A = AddressMode>
  static typename std::enable_if<A == kAddressModeValid, void>::type
  check_sizeof_class() {
    static_assert(AddressMode == kAddressModeValid, "Internal error.");
    static_assert(sizeof(Self) == 1,
                  "SOA class must have only SOA fields.");
  }

  template<int A = AddressMode>
  static typename std::enable_if<A == kAddressModeZero, void>::type
  check_sizeof_class() {
    static_assert(AddressMode == kAddressModeZero, "Internal error.");
    static_assert(sizeof(Self) == 0,
                  "SOA class must have only SOA fields.");
  }

  // Size of this class is 1 if in valid addressing mode. Otherwise, size of
  // this class is 0.
};

#undef IKRA_DEFINE_LAYOUT_FIELD_TYPE

}  // namespace soa
}  // namespace ikra

#endif  // SOA_SOA_H
