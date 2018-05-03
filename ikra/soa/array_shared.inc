 // This implementation is shared by all array classes. B is the base type
 // of the array. The most important functionality in here is interoperability
 // between host and device code, and iterators.

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
  B* device_ptr() {
    return Owner::storage().translate_address_host_to_device(
        reinterpret_cast<B*>(this));
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


private:
template<int OuterVirtualWarpSize, int VirtualWarpSize>
class VirtualWarpRangeArray {
 public:
  // TODO: Consider using pointers instead of array reference + index as
  // internal state.
  class Iterator {
   public:
    __ikra_device__ Iterator(const Self& array_self, IndexType index)
        : array_self_(array_self), index_(index) {}

    __ikra_device__ Iterator& operator++() {    // Prefix increment.
      index_ += OuterVirtualWarpSize/VirtualWarpSize;
      return *this;
    }

    __ikra_device__ B& operator*() const {
      return array_self_[index_];
    }

    __ikra_device__ bool operator==(const Iterator& other) const {
      // TODO: Should also check if array is the same one.
      return index_ == other.index_;
    }

    __ikra_device__ bool operator!=(const Iterator& other) const {
      return !(*this == other);
    }

   private:
    const Self& array_self_;
    IndexType index_;
  };

  __ikra_device__ VirtualWarpRangeArray(const Self& array_self)
      : array_self_(array_self) {
    assert(VirtualWarpSize <= OuterVirtualWarpSize);
  }

  __ikra_device__ Iterator begin() const {
#if __CUDA_ARCH__
    const int tid = threadIdx.x % OuterVirtualWarpSize;
#else
    const int tid = 0;
#endif  // __CUDA_ARCH__
    return Iterator(array_self_, tid/VirtualWarpSize);
  }

  __ikra_device__ Iterator end() const {
#if __CUDA_ARCH__
    const int first_idx =
        (threadIdx.x % OuterVirtualWarpSize)/VirtualWarpSize;
#else
    const int first_idx = 0;
#endif  // __CUDA_ARCH__
    // Formula: R(ArraySize - first_idx) + first_idx
    //          R(i): Round up to next multiple of OuterVWS/VWS
    const int step_size = OuterVirtualWarpSize/VirtualWarpSize;
    const IndexType end_idx =
        ((array_self_.size() - first_idx + step_size - 1)
            /step_size)*step_size + first_idx;
    return Iterator(array_self_, end_idx);
  }

  const Self& array_self_;
};

public:
#ifdef __CUDA_ARCH__
template<int OuterVirtualWarpSize, int VirtualWarpSize = 1>
__ikra_device__ VirtualWarpRangeArray<OuterVirtualWarpSize, VirtualWarpSize>
    vw_iterator() const {
  return VirtualWarpRangeArray<OuterVirtualWarpSize, VirtualWarpSize>(*this);
}
#else
template<int OuterVirtualWarpSize, int VirtualWarpSize = 1>
__ikra_device__ VirtualWarpRangeArray<OuterVirtualWarpSize, VirtualWarpSize>
    vw_iterator() const {
  assert(OuterVirtualWarpSize == 0 && VirtualWarpSize == 1);
  return VirtualWarpRangeArray<1, 1>(*this);
}
#endif

__ikra_device__ typename VirtualWarpRangeArray<1, 1>::Iterator begin() const {
  return VirtualWarpRangeArray<1, 1>::Iterator(*this, 0);
}

__ikra_device__ typename VirtualWarpRangeArray<1, 1>::Iterator end() const {
  return VirtualWarpRangeArray<1, 1>::Iterator(*this, this->size());
}
