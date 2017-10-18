
#ifndef SOA_SOA_H
#define SOA_SOA_H

#include <tuple>
#include <type_traits>

// Asserts active only in debug mode (NDEBUG).
#include <cassert>


namespace ikra {
namespace soa {

#define IKRA_DEFINE_FIELD_ASSIGNMENT(symbol) \
  Self operator symbol ## = (T value) { \
    *data_ptr() symbol ## = value; \
    return *this; \
  }

#define IKRA_DEFINE_LAYOUT_FIELD_TYPE(type) \
  template<int Offset> \
  using type ## _ = Field<type, Offset>;

template<typename T, int ContainerSize, int Offset, class Owner>
class Field_ {
 private:
  using Self = Field_<T, ContainerSize, Offset, Owner>;

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

 protected:
  // Only Owner can create new fields for itself.
  friend Owner;
  Field_() {}

 private:
  Field_(const Field_& other) {}

  // Calculate the address of this field based on the "this" pointer of this
  // Field instance.
  T* data_ptr() const {
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(this)*sizeof(T) +
                                Owner::storage.data + ContainerSize*Offset);
  }

  // Force size of this class to be 0.
  char dummy_[0];
};

#undef IKRA_DEFINE_FIELD_ASSIGNMENT


template<class Self, uint32_t ObjectSizeT, uint32_t ContainerSizeT>
class SoaLayout {
 public:
  void* operator new(size_t count) {
    static_assert(sizeof(Self) == 0,
                  "SOA class must have only SOA fields.");
    assert(count == 0);
    // Check if out of memory.
    assert(Self::storage.size <= ContainerSizeT);

    return reinterpret_cast<Self*>(Self::storage.size++);
  }

  // TODO: Implement delete operator.
  void operator delete(void* ptr) = delete;

  template<uint32_t TotalSize>
  class InternalStorage {
   public:
    uint32_t size;
    char data[TotalSize];
  };

  using Storage = InternalStorage<ObjectSizeT*ContainerSizeT>;

  template<typename T, int Offset>
  using Field = Field_<T, ContainerSizeT, Offset, Self>;

  // Implement more types as necessary.
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(bool);
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(char);
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(double);
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(float);
  IKRA_DEFINE_LAYOUT_FIELD_TYPE(int);

  // Return a pointer to an object with a given ID.
  static Self* get(uintptr_t id) {
    assert(id <= Self::storage.size);
    return reinterpret_cast<Self*>(id);
  }

 private:
  // Force size of this class to be 0.
  char dummy_[0];
};

#undef IKRA_DEFINE_LAYOUT_FIELD_TYPE

}  // namespace soa
}  // namespace ikra

#endif  // SOA_SOA_H
