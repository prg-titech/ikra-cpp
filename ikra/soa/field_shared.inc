// This file contains methods that are common to Field_ and SoaArrayField_.
// Cannot use inheritance here because sizeof cannot be set to 0 in CUDA with
// inheritance.

 protected:
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

  template<int A = AddressMode, int S = StorageMode>
  __ikra_device__ typename std::enable_if<A == kAddressModeZero &&
                                          S == kStorageModeStatic, T*>::type
  data_ptr_uninitialized() const {
    // Use constant-folded value for address computation.
    constexpr auto cptr_data_offset =
        StorageDataOffset<typename Owner::Storage>::value;
    constexpr auto cptr_storage_buffer = Owner::storage_buffer();
    constexpr auto array_location =
        cptr_storage_buffer + cptr_data_offset + Capacity*Offset;

#ifdef __clang__
    // Clang does not array reinterpret_cast in constexprs.
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
        p_base + Capacity*Offset +
        reinterpret_cast<uintptr_t>(this)*sizeof(T));
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