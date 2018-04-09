  // This file should be included for field types that have an address. E.g.,
  // AOS-style arrays have a single address, but SOA-style arrays do not.
  
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
    auto h_data_ptr = reinterpret_cast<uintptr_t>(data_ptr());
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