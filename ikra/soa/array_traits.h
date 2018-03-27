#ifndef SOA_ARRAY_TRAITS_H
#define SOA_ARRAY_TRAITS_H

#include "soa/constants.h"

namespace ikra {
namespace soa {

// Code adapted from:
// https://stackoverflow.com/questions/45801696/no-type-named-type-in-ctrp-derived-class
template<typename T> struct ArrayTraits;

template<typename ElementType, size_t ArraySize> class Array;
template<typename ElementType, size_t ArraySize>
struct ArrayTraits<Array<ElementType, ArraySize>> {
  using T = ElementType;
  static const int N = ArraySize;
};

template<typename ElementType, size_t ArraySize, IndexType Capacity,
         uint32_t Offset, int AddressMode, int StorageMode, class Owner>
class AosArrayField_;
template<typename ElementType, size_t ArraySize, IndexType Capacity,
         uint32_t Offset, int AddressMode, int StorageMode, class Owner>
struct ArrayTraits<AosArrayField_<ElementType, ArraySize, Capacity,
                  Offset,AddressMode, StorageMode, Owner>> {
  using T = ElementType;
  static const int N = ArraySize;
};

template<typename ElementType, size_t ArraySize, IndexType Capacity,
         uint32_t Offset, int AddressMode, int StorageMode, class Owner>
class SoaArrayField_;
template<typename ElementType, size_t ArraySize, IndexType Capacity,
         uint32_t Offset, int AddressMode, int StorageMode, class Owner>
struct ArrayTraits<SoaArrayField_<ElementType, ArraySize, Capacity,
                  Offset,AddressMode, StorageMode, Owner>> {
  using T = ElementType;
  static const int N = ArraySize;
};

#define IKRA_DEFINE_TRAITS_OF_EXPRESSION_TEMPLATES(name) \
  template<typename Left, typename Right> class Array##name##_; \
  \
  template<typename Left, typename Right> \
  struct ArrayTraits<Array##name##_<Left, Right>> { \
    using T = typename ArrayTraits<Left>::T; \
    static const int N = ArrayTraits<Left>::N; \
  };

IKRA_DEFINE_TRAITS_OF_EXPRESSION_TEMPLATES(Add);
IKRA_DEFINE_TRAITS_OF_EXPRESSION_TEMPLATES(Sub);
IKRA_DEFINE_TRAITS_OF_EXPRESSION_TEMPLATES(Mul);
IKRA_DEFINE_TRAITS_OF_EXPRESSION_TEMPLATES(Div);
IKRA_DEFINE_TRAITS_OF_EXPRESSION_TEMPLATES(Mod);
IKRA_DEFINE_TRAITS_OF_EXPRESSION_TEMPLATES(ScalarAdd);
IKRA_DEFINE_TRAITS_OF_EXPRESSION_TEMPLATES(ScalarSub);
IKRA_DEFINE_TRAITS_OF_EXPRESSION_TEMPLATES(ScalarMul);
IKRA_DEFINE_TRAITS_OF_EXPRESSION_TEMPLATES(ScalarDiv);
IKRA_DEFINE_TRAITS_OF_EXPRESSION_TEMPLATES(ScalarMod);

}  // namespace soa
}  // namespace ikra

#endif  // SOA_ARRAY_TRAITS_H
