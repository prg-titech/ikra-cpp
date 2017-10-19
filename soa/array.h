#ifndef SOA_ARRAY_H
#define SOA_ARRAY_H

namespace ikra {
namespace soa {
namespace {

template<typename T,
         uintptr_t ContainerSize,
         uint32_t Offset,
         int AddressMode,
         class Owner>
class AosArrayField_ : public Field_<T, ContainerSize, Offset,
                                     AddressMode, Owner> {
 public:
  // This operator is just for convenience reasons. The correct way to use it
  // would be "this->operator[](pos)".
  typename T::reference operator[](typename T::size_type pos) {
    return this->data_ptr()->operator[](pos);
  }
};

}  // namespace
}  // namespace soa
}  // namespace ikra

#endif  // SOA_ARRAY_H
