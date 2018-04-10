 // This implementation is shared by all array classes. B is the base type
 // of the array. The most important functionality in here is interoperability
 // between host and device code.

public:
// Implement std::array interface.

#if defined(__CUDA_ARCH__) || !defined(__CUDACC__)
// Not running in CUDA mode or running on device: Can return a reference to
// array values.

__ikra_device__ B& operator[](size_t pos) const {
  return *this->array_data_ptr(pos);
}

__ikra_device__ B& at(size_t pos) const {
  // TODO: This function should throw an exception.
  assert(pos < ArraySize);
  return this->operator[](pos);
}

template<size_t Pos>
__ikra_device__ B& at() const {
  static_assert(Pos < ArraySize, "Index out of bounds.");
  return *array_data_ptr<Pos>();
}

__ikra_device__ B& front() const {
  return at<0>();
}

__ikra_device__ B& back() const {
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
  void copy_from_device(B* target) {
    auto dev_ptr = device_ptr();
    cudaMemcpy(target, dev_ptr, sizeof(B), cudaMemcpyDeviceToHost);
    assert(cudaPeekAtLastError() == cudaSuccess);
  }

  B copy_from_device() {
    B host_data;
    copy_from_device(&host_data);
    return host_data;
  }

  // Implicit conversion: Copy from device.
  operator B() {
    return copy_from_device();
  }

  // TODO: Assuming zero addressing mode.
  AssignmentHelper& operator=(B value) {
    cudaMemcpy(device_ptr(), &value, sizeof(B), cudaMemcpyHostToDevice);
    assert(cudaPeekAtLastError() == cudaSuccess);
    return *this;
  }

  AssignmentHelper& operator+=(B value) {
    // TODO: Implement.
    printf("Warning: Calling unimplemented function AssignmentHelper+=.\n");
    assert(false);
  }

  B operator->() {
    return copy_from_device();
  }

 private:
  B* device_ptr() const {
    auto h_data_ptr = reinterpret_cast<uintptr_t>(this);
    auto h_storage_ptr = reinterpret_cast<uintptr_t>(&Owner::storage());
    assert(h_data_ptr >= h_storage_ptr);
    auto data_offset = h_data_ptr - h_storage_ptr;
    auto d_storage_ptr = reinterpret_cast<uintptr_t>(
        Owner::storage().device_ptr());
    return reinterpret_cast<B*>(d_storage_ptr + data_offset);
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
