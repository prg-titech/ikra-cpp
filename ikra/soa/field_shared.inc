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

template<int A = AddressMode, int L = LayoutMode>
__ikra_device__
typename std::enable_if<A == kAddressModeZero, IndexType>::type
id() const {
  static_assert(sizeof(Self) == 0,
                "Zero addressing mode not supported by compiler.");
  // TODO: Check if there's a slowdown here due to forwarding this.
  return reinterpret_cast<Owner*>(this)->id();
}

// Force size of this class to be 0.
char dummy_[0];

