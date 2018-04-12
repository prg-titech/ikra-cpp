  // This file should be included for field types that have an address. E.g.,
  // AOS-style arrays have a single address, but SOA-style arrays do not.

public:
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

protected:
// Calculate the address of this field based on the "this" pointer of this
// Field instance.
template<int A = AddressMode>
__ikra_device__ typename std::enable_if<A != kAddressModeZero, T*>::type
data_ptr_uninitialized() const {
  auto p_this = reinterpret_cast<uintptr_t>(this);
  auto p_base = reinterpret_cast<uintptr_t>(Owner::storage().data_ptr());
  auto p_result = (p_this - p_base - A)/A*sizeof(T) + p_base +
                   (Capacity+1)*Offset;
  return reinterpret_cast<T*>(p_result);
}

template<int A = AddressMode>
__ikra_device__ typename std::enable_if<A != kAddressModeZero, T*>::type
data_ptr() const {
  return data_ptr_uninitialized();
}

template<int A = AddressMode, int S = StorageMode>
__ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                        S == kStorageModeStatic, T*>::type
data_ptr_uninitialized() const {
  // Use constant-folded value for address computation.
  constexpr auto cptr_data_offset =
      StorageDataOffset<typename Owner::Storage>::value;
  constexpr auto cptr_storage_buffer = Owner::storage_buffer();
  constexpr auto array_location =
      cptr_storage_buffer + cptr_data_offset + (Capacity+1)*Offset;

#ifdef __clang__
  // Clang does not allow reinterpret_cast in constexprs.
  constexpr T* soa_array = IKRA_fold(reinterpret_cast<T*>(array_location));
#else
  constexpr T* soa_array = reinterpret_cast<T*>(array_location);
#endif  // __clang__

  // Check alignment.
  assert(reinterpret_cast<uintptr_t>(soa_array) % 8 == 0);

  return soa_array + reinterpret_cast<uintptr_t>(this);
}

template<int A = AddressMode, int S = StorageMode>
__ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                        S == kStorageModeDynamic, T*>::type
data_ptr_uninitialized() const {
  // Cannot constant fold dynamically allocated storage.
  auto p_base = Owner::storage().data_reference();
  return reinterpret_cast<T*>(
      p_base + (Capacity+1)*Offset +
      reinterpret_cast<uintptr_t>(this)*sizeof(T));
}

template<int A = AddressMode>
__ikra_device__ typename std::enable_if<A == kAddressModeZero, T*>::type
data_ptr() const {
  return data_ptr_uninitialized();
}
