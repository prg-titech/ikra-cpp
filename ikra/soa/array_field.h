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
         IndexType Capacity,
         uint32_t Offset,
         int AddressMode,
         class Owner>
class AosArrayField_ : public Field_<T, Capacity, Offset,
                                     AddressMode, Owner> {
 public:
  static const int kSize = sizeof(T);

  // This operator is just for convenience reasons. The correct way to use it
  // would be "this->operator[](pos)".
  typename T::reference operator[](typename T::size_type pos) {
    return this->data_ptr()->operator[](pos);
  }

  operator T&() const {
    return *this->data_ptr();
  }

#include "soa/field_shared.inc"
};

// Class for field declarations of type array. T is the base type of the array.
// This class is the SoA counter part of AosArrayField_. Array slots are
// layouted as if they were SoA fields (columns).
template<typename T,
         size_t ArraySize,
         IndexType Capacity,
         uint32_t Offset,
         int AddressMode,
         class Owner>
class SoaArrayField_ {
 private:
  using Self = SoaArrayField_<T, ArraySize, Capacity, Offset,
                              AddressMode, Owner>;

 public:
  static const int kSize = sizeof(std::array<T, ArraySize>);

  // Support calling methods using -> syntax.
  __ikra_device__ const Self* operator->() const {
    return this;
  }

  T* operator&() const  = delete;

  T& get() const = delete;

  operator T&() const = delete;

  // Implement std::array interface.

#if defined(__CUDA_ARCH__) || !defined(__CUDACC__)
  // Not running in CUDA mode or running on device: Can return a reference to
  // array values.

  __ikra_device__ T& operator[](size_t pos) const {
    return *this->array_data_ptr(pos);
  }

  __ikra_device__ T& at(size_t pos) const {
    // TODO: This function should throw an exception.
    assert(pos < ArraySize);
    return this->operator[](pos);
  }

  template<size_t Pos>
  __ikra_device__ T& at() const {
    static_assert(Pos < ArraySize, "Index out of bounds.");
    return *array_data_ptr<Pos>();
  }

  __ikra_device__ T& front() const {
    return at<0>();
  }

  __ikra_device__ T& back() const {
    return at<ArraySize - 1>();
  }
#else

  // A helper class with an overridden operator= method. This class allows
  // clients to the "[]=" syntax for array assignment, even if the array is
  // physically located on the device.
  // Addresses of instances point to host data locations.
  class AssignmentHelper {
   public:
    // TODO: Assuming zero addressing mode. Must translate addresses in valid
    // addressing mode.
    void copy_from_device(T* target) {
      cudaMemcpy(target, device_ptr(), sizeof(T), cudaMemcpyDeviceToHost);
      assert(cudaPeekAtLastError() == cudaSuccess);
    }

    T copy_from_device() {
      T host_data;
      copy_from_device(&host_data);
      return host_data;
    }

    // Implicit conversion: Copy from device.
    operator T() {
      return copy_from_device();
    }

    // TODO: Assuming zero addressing mode.
    AssignmentHelper& operator=(T value) {
      cudaMemcpy(device_ptr(), &value, sizeof(T), cudaMemcpyHostToDevice);
      assert(cudaPeekAtLastError() == cudaSuccess);
      return *this;
    }

    AssignmentHelper& operator+=(T value) {
      // TODO: Implement.
      printf("Warning: Calling unimplemented function AssignmentHelper+=.\n");
      assert(false);
    }

    T operator->() {
      return copy_from_device();
    }

   private:
    T* device_ptr() {
      auto h_data_ptr = reinterpret_cast<uintptr_t>(this);
      auto h_storage_ptr = reinterpret_cast<uintptr_t>(&Owner::storage());
      assert(h_data_ptr >= h_storage_ptr);
      auto data_offset = h_data_ptr - h_storage_ptr;
      auto d_storage_ptr = reinterpret_cast<uintptr_t>(
          Owner::storage().device_ptr());
      return reinterpret_cast<T*>(d_storage_ptr + data_offset);
    }
  };

  AssignmentHelper& operator[](size_t pos) const {
    return *reinterpret_cast<AssignmentHelper*>(array_data_ptr(pos));
  }

  AssignmentHelper& at(size_t pos) const {
    // TODO: This function should throw an exception.
    assert(pos < ArraySize);
    return this->operator[](pos);
  }

  // TODO: Implement template-based accessor methods.
#endif

  // TODO: Implement iterator and other methods.

 protected:
  // Calculate the address of an array element. For details, see comment
  // of data_ptr in Field_.
  template<size_t Pos, int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, T*>::type
  array_data_ptr() const {
    // Ensure that this is a valid pointer: Only those objects may be accessed
    // which were created with the "new" keyword and are thus initialized.
    assert(this->id() < Owner::storage().size());

    auto p_this = reinterpret_cast<uintptr_t>(this);
    auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());
    auto p_result = (p_this - p_base - A)/A*sizeof(T) + p_base +
                    Capacity*(Offset + Pos*sizeof(T));
    return reinterpret_cast<T*>(p_result);
  }

  template<size_t Pos, int A = AddressMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero, T*>::type
  array_data_ptr() const {
    assert(this->id() < Owner::storage().size());

#ifndef __CUDACC__
    // No constant folding fix in CUDA mode.
    if (Owner::Storage::kIsStaticStorage) {
      // Use constant-folded value for address computation.
      constexpr uintptr_t cptr_data_offset =
          StorageDataOffset<typename Owner::Storage>::value;
      constexpr char* cptr_storage_buffer =
          IKRA_fold(reinterpret_cast<char*>(Owner::storage_buffer()));
      constexpr char* array_location =
          cptr_storage_buffer + cptr_data_offset +
          Capacity*(Offset + Pos*sizeof(T));
      constexpr T* soa_array = IKRA_fold(reinterpret_cast<T*>(array_location));

      return soa_array + reinterpret_cast<uintptr_t>(this);
    } else
#endif  // __CUDACC__
    {
      // Cannot constant fold dynamically allocated storage.
      auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());
      return reinterpret_cast<T*>(
          reinterpret_cast<uintptr_t>(this)*sizeof(T) +
          p_base + Capacity*(Offset + Pos*sizeof(T)));
    }
  }

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A != kAddressModeZero, T*>::type
  array_data_ptr(size_t pos) const {
    // Ensure that this is a valid pointer: Only those objects may be accessed
    // which were created with the "new" keyword and are thus initialized.
    assert(this->id() < Owner::storage().size());

    auto p_this = reinterpret_cast<uintptr_t>(this);
    auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());
    auto p_result = (p_this - p_base - A)/A*sizeof(T) + p_base +
                    Capacity*(Offset + pos*sizeof(T));
    return reinterpret_cast<T*>(p_result);
  }

  template<int A = AddressMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero, T*>::type
  array_data_ptr(size_t pos) const {
    assert(this->id() < Owner::storage().size());

#ifndef __CUDACC__
    // No constant folding fix in CUDA mode.
    if (Owner::Storage::kIsStaticStorage) {
      // Use constant-folded value for address computation.
      constexpr uintptr_t cptr_data_offset =
          StorageDataOffset<typename Owner::Storage>::value;
      constexpr char* cptr_storage_buffer =
          IKRA_fold(reinterpret_cast<char*>(Owner::storage_buffer()));
      constexpr char* array_location =
          cptr_storage_buffer + cptr_data_offset + Capacity*Offset;
      T* soa_array = reinterpret_cast<T*>(array_location + pos*sizeof(T)*Capacity);

      return soa_array + reinterpret_cast<uintptr_t>(this);
    } else
#endif  // __CUDACC__
    {
      // Cannot constant fold dynamically allocated storage.
      auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());
      return reinterpret_cast<T*>(
          reinterpret_cast<uintptr_t>(this)*sizeof(T) +
          p_base + Capacity*(Offset + pos*sizeof(T)));
    }
  }

#include "soa/field_shared.inc"
};

}  // namespace soa
}  // namespace ikra

#endif  // SOA_ARRAY_H
