#ifndef SOA_ARRAY_EXPRESSION_H
#define SOA_ARRAY_EXPRESSION_H

#include <type_traits>

#include "soa/array_traits.h"
#include "soa/util.h"

namespace ikra {
namespace soa {

// Superclass for all kinds of arrays (e.g., SoaArrayField_, ArrayAdd_).
// `Self` is the type of the derived class (Curiously Recurring Template Pattern).
template<typename Self>
class ArrayExpression_ {
 private:
  using T = typename ArrayTraits<Self>::T;
  static const int N = ArrayTraits<Self>::N;

 public:
  // Force size of this class to be 0.
  char dummy_[0];

  static const bool kIsArrayExpression = true;

  T operator[](size_t i) const {
    return static_cast<Self const&>(*this)[i];
  }
  T& operator[](size_t i) {
    return static_cast<Self&>(*this)[i];
  }
  size_t size() const { return N; }

  operator Self&() {
    return static_cast<Self&>(*this);
  }
  operator Self const&() {
    return static_cast<Self const&>(*this);
  }
};

// These macros are used to overload assignment operators for array expressions.
// ElementType and ArraySize should be defined within the context.
#define IKRA_DEFINE_ARRAY_ASSIGNMENT(op) \
  template<typename Self> \
  void operator op##=(ArrayExpression_<Self> const& a) { \
    for (size_t i = 0; i < ArraySize; ++i) { \
      (*this)[i] op##= a[i]; \
    } \
  }
#define IKRA_DEFINE_ARRAY_SCALAR_ASSIGNMENT(op) \
  void operator op##=(ElementType c) { \
    for (size_t i = 0; i < ArraySize; ++i) { \
      (*this)[i] op##= c; \
    } \
  }

// A wrapper for std::array to make it inherit from ArrayExpression_.
template<typename ElementType, size_t ArraySize>
class Array : public ArrayExpression_<Array<ElementType, ArraySize>> {
 private:
  std::array<ElementType, ArraySize> array_;

 public:
  ElementType operator[](size_t i) const { return array_[i]; }
  ElementType& operator[](size_t i) { return array_[i]; }

  Array() = default;
  Array(std::array<ElementType, ArraySize> a) : array_(a) {}

  // An array wrapper can be constructed from any ArrayExpression_,
  // which forces its evaluation.
  template<typename Self>
  Array(ArrayExpression_<Self> const& a) {
    static_assert(std::is_same<ElementType, typename ArrayTraits<Self>::T>::value,
        "The specified element type is wrong.");
    static_assert(ArraySize == ArrayTraits<Self>::N,
        "The specified array size is wrong.");

    for (size_t i = 0; i < ArraySize; ++i) {
      array_[i] = a[i];
    }
  }

  // Define the assignment operator between array wrappers and array expressions.
  IKRA_DEFINE_ARRAY_ASSIGNMENT();
  // Defines compound assignment operators between array wrappers and array expressions.
  IKRA_APPLY_TO_ALL_OPERATORS(IKRA_DEFINE_ARRAY_ASSIGNMENT);
  // Defines compound assignment operators between array wrappers and scalars.
  IKRA_APPLY_TO_ALL_OPERATORS(IKRA_DEFINE_ARRAY_SCALAR_ASSIGNMENT);
};

// Generates classes and operators for the operations bewteen array expressions.
// `name` will be part of the class name (e.g., Add -> ArrayAdd_),
// and `op` is the corresponding operator.
#define IKRA_DEFINE_ARRAY_OPERATION(name, op) \
  template<typename Left, typename Right> \
  class Array##name##_ : public ArrayExpression_<Array##name##_<Left, Right>> { \
   private: \
    Left  const& left_; \
    Right const& right_; \
  \
    using T = typename ArrayTraits<Left>::T; \
    static const int N = ArrayTraits<Left>::N; \
  \
   public: \
    Array##name##_(Left const& l, Right const& r) : left_(l), right_(r) { \
      static_assert(Left::kIsArrayExpression, \
          "The left operand is not an array expression."); \
      static_assert(Right::kIsArrayExpression, \
          "The right operand is not an array expression."); \
      static_assert(std::is_same<T, typename ArrayTraits<Right>::T>::value, \
          "The element type of two operands are not the same."); \
      static_assert(N == ArrayTraits<Right>::N, \
          "The array sizes of two operands are not the same."); \
    }; \
    T operator[](size_t i) const { \
      return left_[i] op right_[i]; \
    } \
  }; \
  \
  template<typename Left, typename Right, \
           bool = Left::kIsArrayExpression, \
           bool = Right::kIsArrayExpression> \
  Array##name##_<Left, Right> operator op(Left const& l, Right const& r) { \
    return Array##name##_<Left, Right>(l, r); \
  }

// Generates classes and operators for the scalar operation of array expressions.
// The left template argument is an array while the right one is a scalar.
// `name` will be part of the class name (e.g., Mul -> ArrayScalarMul_),
// and `op` is the corresponding operator.
#define IKRA_DEFINE_ARRAY_SCALAR_OPERATION(name, op) \
  template<typename Left, typename Right> \
  class ArrayScalar##name##_ : \
      public ArrayExpression_<ArrayScalar##name##_<Left, Right>> { \
   private: \
    Left const& left_; \
    Right right_; \
  \
   public: \
    ArrayScalar##name##_(Left const& l, Right r) : left_(l), right_(r) { \
      static_assert(Left::kIsArrayExpression, \
          "The left operand is not an array expression."); \
      static_assert(std::is_same<typename ArrayTraits<Left>::T, Right>::value, \
          "The element type of two operands are not the same."); \
    } \
    Right operator[](size_t i) const { \
      return left_[i] op right_; \
    } \
  }; \
  \
  template<typename Left, typename Right, bool = Left::kIsArrayExpression> \
  typename std::enable_if< \
    std::is_same<typename ArrayTraits<Left>::T, Right>::value, \
    ArrayScalar##name##_<Left, Right> \
  >::type operator op(Left const& l, Right r) { \
    return ArrayScalar##name##_<Left, Right>(l, r); \
  } \
  template<typename Left, typename Right, bool = Right::kIsArrayExpression> \
  typename std::enable_if< \
    std::is_same<typename ArrayTraits<Right>::T, Left>::value, \
    ArrayScalar##name##_<Right, Left> \
  >::type operator op(Left l, Right const& r) { \
    return ArrayScalar##name##_<Right, Left>(r, l); \
  }

IKRA_DEFINE_ARRAY_OPERATION(Add, +);
IKRA_DEFINE_ARRAY_OPERATION(Sub, -);
IKRA_DEFINE_ARRAY_OPERATION(Mul, *);
IKRA_DEFINE_ARRAY_OPERATION(Div, /);
IKRA_DEFINE_ARRAY_OPERATION(Mod, %);
IKRA_DEFINE_ARRAY_SCALAR_OPERATION(Add, +);
IKRA_DEFINE_ARRAY_SCALAR_OPERATION(Sub, -);
IKRA_DEFINE_ARRAY_SCALAR_OPERATION(Mul, *);
IKRA_DEFINE_ARRAY_SCALAR_OPERATION(Div, /);
IKRA_DEFINE_ARRAY_SCALAR_OPERATION(Mod, %);

}  // namespace soa
}  // namespace ikra

#endif  // SOA_ARRAY_EXPRESSION_H
