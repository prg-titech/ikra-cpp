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

// Force size of this class to be 0.
char dummy_[0];