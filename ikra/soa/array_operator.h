#ifndef SOA_ARRAY_OPERATOR_H
#define SOA_ARRAY_OPERATOR_H

#include "soa/array_field.h"
#include "soa/util.h"

#define TEMPLATE_ARRAY_FIELD \
  template<typename T, size_t ArraySize, \
           IndexType Capacity, uint32_t Offset, \
           int AddressMode, typename Owner>
#define TEMPLATE_ARRAY_FIELD_W \
  template<typename T, size_t ArraySize, \
           IndexType Capacity1, uint32_t Offset1, \
           int AddressMode1, typename Owner1, \
           IndexType Capacity2, uint32_t Offset2, \
           int AddressMode2, typename Owner2>
#define AOS_ARRAY_FIELD(n) AosArrayField_<std::array<T, ArraySize>, \
  Capacity ## n, Offset ## n, AddressMode ## n, Owner ## n>
#define SOA_ARRAY_FIELD(n) SoaArrayField_<T, ArraySize, \
  Capacity ## n, Offset ## n, AddressMode ## n, Owner ## n>
#define STD_ARRAY std::array<T, ArraySize>

#define IKRA_DEFINE_STD_ARRAY_OPERATORS(symbol) \
  template<typename T, size_t N> \
  std::array<T, N> operator symbol (std::array<T, N>& left, \
                                    std::array<T, N>& right) { \
    std::array<T, N> result; \
    for (size_t i = 0; i < N; ++i) { \
      result[i] = left[i] symbol right[i]; \
    } \
    return result; \
  } \
  TEMPLATE_ARRAY_FIELD \
  STD_ARRAY operator symbol (STD_ARRAY& left, \
                             AOS_ARRAY_FIELD()& right) { \
    STD_ARRAY result; \
    for (size_t i = 0; i < ArraySize; ++i) { \
      result[i] = left[i] symbol right[i]; \
    } \
    return result; \
  } \
  TEMPLATE_ARRAY_FIELD \
  STD_ARRAY operator symbol (STD_ARRAY& left, \
                             SOA_ARRAY_FIELD()& right) { \
    STD_ARRAY result; \
    for (size_t i = 0; i < ArraySize; ++i) { \
      result[i] = left[i] symbol right[i]; \
    } \
    return result; \
  } \
  template<typename T, size_t N> \
  void operator symbol ## = (std::array<T, N>& left, \
                             std::array<T, N>& right) { \
    for (size_t i = 0; i < N; ++i) { \
      left[i] symbol ## = right[i]; \
    } \
  } \
  TEMPLATE_ARRAY_FIELD \
  void operator symbol ## = (STD_ARRAY& left, \
                             AOS_ARRAY_FIELD()& right) { \
    for (size_t i = 0; i < ArraySize; ++i) { \
      left[i] symbol ## = right[i]; \
    } \
  } \
  TEMPLATE_ARRAY_FIELD \
  void operator symbol ## = (STD_ARRAY& left, \
                             SOA_ARRAY_FIELD()& right) { \
    for (size_t i = 0; i < ArraySize; ++i) { \
      left[i] symbol ## = right[i]; \
    } \
  }

#define IKRA_DEFINE_AOS_ARRAY_FIELD_OPERATORS(symbol) \
  TEMPLATE_ARRAY_FIELD \
  STD_ARRAY operator symbol (AOS_ARRAY_FIELD()& left, \
                             STD_ARRAY& right) { \
    return (STD_ARRAY&)left symbol right; \
  } \
  TEMPLATE_ARRAY_FIELD_W \
  STD_ARRAY operator symbol (AOS_ARRAY_FIELD(1)& left, \
                             AOS_ARRAY_FIELD(2)& right) { \
    return (STD_ARRAY&)left symbol (STD_ARRAY&)right; \
  } \
  TEMPLATE_ARRAY_FIELD_W \
  STD_ARRAY operator symbol (AOS_ARRAY_FIELD(1)& left, \
                             SOA_ARRAY_FIELD(2)& right) { \
    return (STD_ARRAY&)left symbol right; \
  } \
  TEMPLATE_ARRAY_FIELD \
  void operator symbol ## = (AOS_ARRAY_FIELD()& left, \
                             STD_ARRAY& right) { \
    (STD_ARRAY&)left symbol ## = right; \
  } \
  TEMPLATE_ARRAY_FIELD_W \
  void operator symbol ## = (AOS_ARRAY_FIELD(1)& left, \
                             AOS_ARRAY_FIELD(2)& right) { \
    (STD_ARRAY&)left symbol ## = (STD_ARRAY&)right; \
  } \
  TEMPLATE_ARRAY_FIELD_W \
  void operator symbol ## = (AOS_ARRAY_FIELD(1)& left, \
                             SOA_ARRAY_FIELD(2)& right) { \
    (STD_ARRAY&)left symbol ## = right; \
  }

#define IKRA_DEFINE_SOA_ARRAY_FIELD_OPERATORS(symbol) \
  TEMPLATE_ARRAY_FIELD \
  STD_ARRAY operator symbol (SOA_ARRAY_FIELD()& left, \
                             STD_ARRAY& right) { \
    STD_ARRAY result; \
    for (size_t i = 0; i < ArraySize; ++i) { \
      result[i] = left[i] symbol right[i]; \
    } \
    return result; \
  } \
  TEMPLATE_ARRAY_FIELD_W \
  STD_ARRAY operator symbol (SOA_ARRAY_FIELD(1)& left, \
                             AOS_ARRAY_FIELD(2)& right) { \
    STD_ARRAY result; \
    for (size_t i = 0; i < ArraySize; ++i) { \
      result[i] = left[i] symbol right[i]; \
    } \
    return result; \
  } \
  TEMPLATE_ARRAY_FIELD_W \
  STD_ARRAY operator symbol (SOA_ARRAY_FIELD(1)& left, \
                             SOA_ARRAY_FIELD(2)& right) { \
    STD_ARRAY result; \
    for (size_t i = 0; i < ArraySize; ++i) { \
      result[i] = left[i] symbol right[i]; \
    } \
    return result; \
  } \
  TEMPLATE_ARRAY_FIELD \
  void operator symbol ## = (SOA_ARRAY_FIELD()& left, \
                             STD_ARRAY& right) { \
    for (size_t i = 0; i < ArraySize; ++i) { \
      left[i] symbol ## = right[i]; \
    } \
  } \
  TEMPLATE_ARRAY_FIELD_W \
  void operator symbol ## = (SOA_ARRAY_FIELD(1)& left, \
                             AOS_ARRAY_FIELD(2)& right) { \
    for (size_t i = 0; i < ArraySize; ++i) { \
      left[i] symbol ## = right[i]; \
    } \
  } \
  TEMPLATE_ARRAY_FIELD_W \
  void operator symbol ## = (SOA_ARRAY_FIELD(1)& left, \
                             SOA_ARRAY_FIELD(2)& right) { \
    for (size_t i = 0; i < ArraySize; ++i) { \
      left[i] symbol ## = right[i]; \
    } \
  }

namespace ikra {
namespace soa {

IKRA_DEFINE_STD_ARRAY_OPERATORS(+)
IKRA_DEFINE_STD_ARRAY_OPERATORS(-)
IKRA_DEFINE_AOS_ARRAY_FIELD_OPERATORS(+)
IKRA_DEFINE_AOS_ARRAY_FIELD_OPERATORS(-)
IKRA_DEFINE_SOA_ARRAY_FIELD_OPERATORS(+)
IKRA_DEFINE_SOA_ARRAY_FIELD_OPERATORS(-)

}  // soa
}  // ikra

#undef TEMPLATE_ARRAY_FIELD
#undef TEMPLATE_ARRAY_FIELD_W
#undef AOS_ARRAY_FIELD
#undef SOA_ARRAY_FIELD
#undef STD_ARRAY
#undef IKRA_DEFINE_STD_ARRAY_OPERATORS
#undef IKRA_DEFINE_AOS_ARRAY_FIELD_OPERATORS
#undef IKRA_DEFINE_SOA_ARRAY_FIELD_OPERATORS

#endif  // SOA_ARRAY_OPERATOR_H
