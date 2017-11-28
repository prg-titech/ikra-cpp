#ifndef SOA_ARRAY_OPERATOR_H
#define SOA_ARRAY_OPERATOR_H

#include "soa/array_field.h"
#include "soa/util.h"

// This is used for operations between std::arrays and array fields.
#define IKRA_TEMPLATE_ARRAY_FIELD \
  template<typename T, size_t ArraySize, \
           IndexType Capacity, uint32_t Offset, \
           int AddressMode, int StorageMode, typename Owner>
// This is used for operations between two array fields.
#define IKRA_TEMPLATE_ARRAY_FIELD_W \
  template<typename T, size_t ArraySize, \
           IndexType Capacity1, uint32_t Offset1, \
           int AddressMode1, int StorageMode1, typename Owner1, \
           IndexType Capacity2, uint32_t Offset2, \
           int AddressMode2, int StorageMode2, typename Owner2>

// For `IKRA_TEMPLATE_ARRAY_FIELD`, just write `XXX_ARRAY_FIELD()`;
// For `IKRA_TEMPLATE_ARRAY_FIELD_W`, write like `XXX_ARRAY_FIELD(1)`.
#define IKRA_AOS_ARRAY_FIELD(n) AosArrayField_<std::array<T, ArraySize>, \
  Capacity ## n, Offset ## n, AddressMode ## n, StorageMode ## n, Owner ## n>
#define IKRA_SOA_ARRAY_FIELD(n) SoaArrayField_<T, ArraySize, \
  Capacity ## n, Offset ## n, AddressMode ## n, StorageMode ## n, Owner ## n>
#define IKRA_STD_ARRAY std::array<T, ArraySize>

// TODO: Provide overload where `left` and `right` are passed as reference.

// Given a symbol, defines the operator and its compound assignment operator
// between std::array and all array representations.
#define IKRA_DEFINE_IKRA_STD_ARRAY_OPERATORS_VECTOR(symbol) \
  template<typename T, size_t N> \
  std::array<T, N> operator symbol (std::array<T, N> left, \
                                    std::array<T, N> right) { \
    std::array<T, N> result; \
    for (size_t i = 0; i < N; ++i) { \
      result[i] = left[i] symbol right[i]; \
    } \
    return result; \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD \
  IKRA_STD_ARRAY operator symbol (IKRA_STD_ARRAY left, \
                                  IKRA_AOS_ARRAY_FIELD()& right) { \
    IKRA_STD_ARRAY result; \
    for (size_t i = 0; i < ArraySize; ++i) { \
      result[i] = left[i] symbol right[i]; \
    } \
    return result; \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD \
  IKRA_STD_ARRAY operator symbol (IKRA_STD_ARRAY left, \
                                  IKRA_SOA_ARRAY_FIELD()& right) { \
    IKRA_STD_ARRAY result; \
    for (size_t i = 0; i < ArraySize; ++i) { \
      result[i] = left[i] symbol right[i]; \
    } \
    return result; \
  } \
  template<typename T, size_t N> \
  void operator symbol ## = (std::array<T, N>& left, \
                             std::array<T, N> right) { \
    for (size_t i = 0; i < N; ++i) { \
      left[i] symbol ## = right[i]; \
    } \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD \
  void operator symbol ## = (IKRA_STD_ARRAY& left, \
                             IKRA_AOS_ARRAY_FIELD()& right) { \
    for (size_t i = 0; i < ArraySize; ++i) { \
      left[i] symbol ## = right[i]; \
    } \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD \
  void operator symbol ## = (IKRA_STD_ARRAY& left, \
                             IKRA_SOA_ARRAY_FIELD()& right) { \
    for (size_t i = 0; i < ArraySize; ++i) { \
      left[i] symbol ## = right[i]; \
    } \
  }


// Given a symbol, defines the operator and its compound assignment operator
// between std::array and a scalar number.
#define IKRA_DEFINE_IKRA_STD_ARRAY_OPERATORS_SCALAR(symbol) \
  template<typename T, size_t N> \
  std::array<T, N> operator symbol (std::array<T, N> left, T right) { \
    std::array<T, N> result; \
    for (size_t i = 0; i < N; ++i) { \
      result[i] = left[i] symbol right; \
    } \
    return result; \
  } \
  template<typename T, size_t N> \
  std::array<T, N> operator symbol (T left, std::array<T, N> right) { \
    std::array<T, N> result; \
    for (size_t i = 0; i < N; ++i) { \
      result[i] = left symbol right[i]; \
    } \
    return result; \
  } \
  template<typename T, size_t N> \
  void operator symbol ## = (std::array<T, N>& left, T right) { \
    for (size_t i = 0; i < N; ++i) { \
      left[i] symbol ## = right; \
    } \
  }

// Given a symbol, defines the operator and its compound assignment operator
// between AOS array field and all array representations.
#define IKRA_DEFINE_IKRA_AOS_ARRAY_FIELD_OPERATORS_VECTOR(symbol) \
  IKRA_TEMPLATE_ARRAY_FIELD \
  IKRA_STD_ARRAY operator symbol (IKRA_AOS_ARRAY_FIELD()& left, \
                                  IKRA_STD_ARRAY right) { \
    return (IKRA_STD_ARRAY&)left symbol right; \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD_W \
  IKRA_STD_ARRAY operator symbol (IKRA_AOS_ARRAY_FIELD(1)& left, \
                                  IKRA_AOS_ARRAY_FIELD(2)& right) { \
    return (IKRA_STD_ARRAY&)left symbol (IKRA_STD_ARRAY&)right; \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD_W \
  IKRA_STD_ARRAY operator symbol (IKRA_AOS_ARRAY_FIELD(1)& left, \
                                  IKRA_SOA_ARRAY_FIELD(2)& right) { \
    return (IKRA_STD_ARRAY&)left symbol right; \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD \
  void operator symbol ## = (IKRA_AOS_ARRAY_FIELD()& left, \
                             IKRA_STD_ARRAY right) { \
    (IKRA_STD_ARRAY&)left symbol ## = right; \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD_W \
  void operator symbol ## = (IKRA_AOS_ARRAY_FIELD(1)& left, \
                             IKRA_AOS_ARRAY_FIELD(2)& right) { \
    (IKRA_STD_ARRAY&)left symbol ## = (IKRA_STD_ARRAY&)right; \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD_W \
  void operator symbol ## = (IKRA_AOS_ARRAY_FIELD(1)& left, \
                             IKRA_SOA_ARRAY_FIELD(2)& right) { \
    (IKRA_STD_ARRAY&)left symbol ## = right; \
  }

// Given a symbol, defines the operator and its compound assignment operator
// between AOS array field and a scalar number.
#define IKRA_DEFINE_IKRA_AOS_ARRAY_FIELD_OPERATORS_SCALAR(symbol) \
  IKRA_TEMPLATE_ARRAY_FIELD \
  IKRA_STD_ARRAY operator symbol (IKRA_AOS_ARRAY_FIELD()& left, T right) { \
    IKRA_STD_ARRAY result; \
    for (size_t i = 0; i < ArraySize; ++i) { \
      result[i] = left[i] symbol right; \
    } \
    return result; \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD \
  IKRA_STD_ARRAY operator symbol (T left, IKRA_AOS_ARRAY_FIELD()& right) { \
    IKRA_STD_ARRAY result; \
    for (size_t i = 0; i < ArraySize; ++i) { \
      result[i] = left symbol right[i]; \
    } \
    return result; \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD \
  void operator symbol ## = (IKRA_AOS_ARRAY_FIELD()& left, T right) { \
    for (size_t i = 0; i < ArraySize; ++i) { \
      left[i] symbol ## = right; \
    } \
  }

// Given a symbol, defines the operator and its compound assignment operator
// between SOA array fields and all array representations.
#define IKRA_DEFINE_IKRA_SOA_ARRAY_FIELD_OPERATORS_VECTOR(symbol) \
  IKRA_TEMPLATE_ARRAY_FIELD \
  IKRA_STD_ARRAY operator symbol (IKRA_SOA_ARRAY_FIELD()& left, \
                                  IKRA_STD_ARRAY right) { \
    IKRA_STD_ARRAY result; \
    for (size_t i = 0; i < ArraySize; ++i) { \
      result[i] = left[i] symbol right[i]; \
    } \
    return result; \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD_W \
  IKRA_STD_ARRAY operator symbol (IKRA_SOA_ARRAY_FIELD(1)& left, \
                                  IKRA_AOS_ARRAY_FIELD(2)& right) { \
    IKRA_STD_ARRAY result; \
    for (size_t i = 0; i < ArraySize; ++i) { \
      result[i] = left[i] symbol right[i]; \
    } \
    return result; \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD_W \
  IKRA_STD_ARRAY operator symbol (IKRA_SOA_ARRAY_FIELD(1)& left, \
                                  IKRA_SOA_ARRAY_FIELD(2)& right) { \
    IKRA_STD_ARRAY result; \
    for (size_t i = 0; i < ArraySize; ++i) { \
      result[i] = left[i] symbol right[i]; \
    } \
    return result; \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD \
  void operator symbol ## = (IKRA_SOA_ARRAY_FIELD()& left, \
                             IKRA_STD_ARRAY right) { \
    for (size_t i = 0; i < ArraySize; ++i) { \
      left[i] symbol ## = right[i]; \
    } \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD_W \
  void operator symbol ## = (IKRA_SOA_ARRAY_FIELD(1)& left, \
                             IKRA_AOS_ARRAY_FIELD(2)& right) { \
    for (size_t i = 0; i < ArraySize; ++i) { \
      left[i] symbol ## = right[i]; \
    } \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD_W \
  void operator symbol ## = (IKRA_SOA_ARRAY_FIELD(1)& left, \
                             IKRA_SOA_ARRAY_FIELD(2)& right) { \
    for (size_t i = 0; i < ArraySize; ++i) { \
      left[i] symbol ## = right[i]; \
    } \
  }

// Given a symbol, defines the operator and its compound assignment operator
// between SOA array field and a scalar number.
#define IKRA_DEFINE_IKRA_SOA_ARRAY_FIELD_OPERATORS_SCALAR(symbol) \
  IKRA_TEMPLATE_ARRAY_FIELD \
  IKRA_STD_ARRAY operator symbol (IKRA_SOA_ARRAY_FIELD()& left, T right) { \
    IKRA_STD_ARRAY result; \
    for (size_t i = 0; i < ArraySize; ++i) { \
      result[i] = left[i] symbol right; \
    } \
    return result; \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD \
  IKRA_STD_ARRAY operator symbol (T left, IKRA_SOA_ARRAY_FIELD()& right) { \
    IKRA_STD_ARRAY result; \
    for (size_t i = 0; i < ArraySize; ++i) { \
      result[i] = left symbol right[i]; \
    } \
    return result; \
  } \
  IKRA_TEMPLATE_ARRAY_FIELD \
  void operator symbol ## = (IKRA_SOA_ARRAY_FIELD()& left, T right) { \
    for (size_t i = 0; i < ArraySize; ++i) { \
      left[i] symbol ## = right; \
    } \
  }

namespace ikra {
namespace soa {

IKRA_DEFINE_ALL_OPERATORS(IKRA_DEFINE_IKRA_STD_ARRAY_OPERATORS_VECTOR)
IKRA_DEFINE_ALL_OPERATORS(IKRA_DEFINE_IKRA_STD_ARRAY_OPERATORS_SCALAR)

IKRA_DEFINE_ALL_OPERATORS(IKRA_DEFINE_IKRA_AOS_ARRAY_FIELD_OPERATORS_VECTOR)
IKRA_DEFINE_ALL_OPERATORS(IKRA_DEFINE_IKRA_AOS_ARRAY_FIELD_OPERATORS_SCALAR)

IKRA_DEFINE_ALL_OPERATORS(IKRA_DEFINE_IKRA_SOA_ARRAY_FIELD_OPERATORS_VECTOR)
IKRA_DEFINE_ALL_OPERATORS(IKRA_DEFINE_IKRA_SOA_ARRAY_FIELD_OPERATORS_SCALAR)

}  // soa
}  // ikra

#undef IKRA_TEMPLATE_ARRAY_FIELD
#undef IKRA_TEMPLATE_ARRAY_FIELD_W
#undef IKRA_AOS_ARRAY_FIELD
#undef IKRA_SOA_ARRAY_FIELD
#undef IKRA_STD_ARRAY
#undef IKRA_DEFINE_IKRA_STD_ARRAY_OPERATORS_VECTOR
#undef IKRA_DEFINE_IKRA_STD_ARRAY_OPERATORS_SCALAR
#undef IKRA_DEFINE_IKRA_AOS_ARRAY_FIELD_OPERATORS_VECTOR
#undef IKRA_DEFINE_IKRA_AOS_ARRAY_FIELD_OPERATORS_SCALAR
#undef IKRA_DEFINE_IKRA_SOA_ARRAY_FIELD_OPERATORS_VECTOR
#undef IKRA_DEFINE_IKRA_SOA_ARRAY_FIELD_OPERATORS_SCALAR

#endif  // SOA_ARRAY_OPERATOR_H
