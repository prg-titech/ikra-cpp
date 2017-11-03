#ifndef SOA_FIELD_H
#define SOA_FIELD_H

#include "soa/constants.h"
#include "soa/cuda.h"
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
         class Owner>
class Field_ {
 private:
  using Self = Field_<T, Capacity, Offset, AddressMode, Owner>;

 public:
  // This class may only be used to declare fields of classes.
  void* operator new(size_t count) = delete;

  // TODO: Disable operator if T is a SOA class. SOA object pointer cannot be
  // dereferenced.
  __ikra_device__ T& operator*() const {
    return *data_ptr();
  }

  __ikra_device__ T* operator->() const {
    return data_ptr();
  }

  __ikra_device__ T* operator&() const {
    return data_ptr();
  }

  // Get the value of this field. This method is not usually needed and the
  // preferred way to retrieve a value is through implicit conversion.
  __ikra_device__ T& get() const {
    return *data_ptr();
  }

#if defined(__CUDA_ARCH__) || !defined(__CUDACC__)
  // Operator for implicit conversion to type T. Either running device code or
  // not running in CUDA mode at all.
  __ikra_device__ operator T&() const {
    return *data_ptr();
  }

  // Assignment operator.
  __ikra_device__ Self& operator=(T value) {
    *data_ptr() = value;
    return *this;
  }
#else
  T* device_data_ptr() const {
    auto h_data_ptr = reinterpret_cast<uintptr_t>(data_ptr_uninitialized());
    auto h_storage_data = reinterpret_cast<uintptr_t>(&Owner::storage());
    auto data_offset = h_data_ptr - h_storage_data;
    auto d_storage_ptr = reinterpret_cast<uintptr_t>(
        Owner::storage().device_ptr());
    return reinterpret_cast<T*>(d_storage_ptr + data_offset);
  }

  void copy_from_device(T* target) const {
    cudaMemcpy(target, device_data_ptr(), sizeof(T), cudaMemcpyDeviceToHost);
  }

  T copy_from_device() const {
    T host_data;
    copy_from_device(&host_data);
    return host_data;
  }

  // Operator for implicit conversion to type T. Running in CUDA mode on the
  // host. Data must be copied.
  // TODO: This method is broken when compiling in CUDA mode but host execution
  // is intended.
  operator T() const {
    return copy_from_device();
  }

  // Assignment operator.
  // TODO: Probably need to handle SOA pointer differently here.
  Self& operator=(T value) {
    cudaMemcpy(device_data_ptr(), &value, sizeof(T), cudaMemcpyHostToDevice);
    return *this;
  }
#endif  // __CUDA_ARCH__

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

  template<int A = AddressMode>
  __ikra_device__
  typename std::enable_if<A != kAddressModeZero, IndexType>::type
  id() const {
    return (reinterpret_cast<uintptr_t>(this) -
            reinterpret_cast<uintptr_t>(Owner::storage().data_ptr()))
               / A - 1 - 1;
  }

  template<int A = AddressMode>
  __ikra_device__
  typename std::enable_if<A == kAddressModeZero, IndexType>::type
  id() const {
    return reinterpret_cast<uintptr_t>(this) - 1;
  }

  // Calculate the address of this field based on the "this" pointer of this
  // Field instance.
  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, T*>::type
  data_ptr_uninitialized() const {
    auto p_this = reinterpret_cast<uintptr_t>(this);
    auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());
    auto p_result = (p_this - p_base - A)/A*sizeof(T) + p_base +
                     Capacity*Offset;
    return reinterpret_cast<T*>(p_result);
  }

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, T*>::type
  data_ptr() const {
    // Ensure that this is a valid pointer: Only those objects may be accessed
    // which were created with the "new" keyword and are thus initialized.
    assert(id() < Owner::storage().size());
    return data_ptr_uninitialized();
  }

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero, T*>::type
  data_ptr_uninitialized() const {
    auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(this)*sizeof(T) +
                                p_base + Capacity*Offset);
  }

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero, T*>::type
  data_ptr() const {
    // Ensure that this is a valid pointer: Only those objects may be accessed
    // which were created with the "new" keyword and are thus initialized.
    assert(id() < Owner::storage().size());
    return data_ptr_uninitialized();
  }

  // Force size of this class to be 0.
  char dummy_[0];
};

}  // namespace soa
}  // namespace ikra

#undef IKRA_DEFINE_FIELD_ASSIGNMENT

#endif  // SOA_FIELD_H
