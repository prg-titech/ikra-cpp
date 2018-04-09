#ifndef SOA_FIELD_H
#define SOA_FIELD_H

#include "soa/constants.h"
#include "soa/cuda.h"
#include "soa/storage.h"
#include "soa/util.h"

// This marco is expanded for every inplace assignment operator and forwards
// the operation to the wrapped data.
#define IKRA_DEFINE_FIELD_ASSIGNMENT(symbol) \
  __ikra_device__ Self operator symbol ## = (T value) { \
    *data_ptr() symbol ## = value; \
    return *this; \
  }


namespace ikra {
namespace soa {

template<typename T,
         IndexType Capacity,
         uint32_t Offset,
         int AddressMode,
         int StorageMode,
         class Owner>
class Field_ {
 private:
  using Self = Field_<T, Capacity, Offset, AddressMode, StorageMode, Owner>;

 public:
  static const uint32_t DBG_OFFSET = Offset;  // For debugging.

  // This class may only be used to declare fields of classes.
  void* operator new(size_t count) = delete;

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
#if defined(__CUDA_ARCH__) || !defined(__CUDACC__)
  // Only Owner can create new fields for itself.
  friend Owner;
#else
  // Only Owner can create new fields for itself.
  // Friend template is broken. This is an nvcc bug.
  friend typename NvccWorkaroundIdentityClassHolder<Owner>::type;
#endif

  __ikra_device__ Field_() {}

  // Initialize the field with a given value. Used in class constructor.
  __ikra_device__ Field_(T value) {
    *this = value;
  }

  // Initialize the field with the value of another field. Used in class
  // constructor. TODO: Unclear why exactly we need this one...
  __ikra_device__ Field_(const Field_& /*other*/) {}

  // Not sure why we need this. Required to make field inplace initialization
  // work.
  __ikra_device__ Field_(Field_&& /*other*/) {}


#include "soa/field_shared.inc"
#include "soa/addressable_field_shared.inc"
};

}  // namespace soa
}  // namespace ikra

#undef IKRA_DEFINE_FIELD_ASSIGNMENT

#endif  // SOA_FIELD_H
