#ifndef EXECUTOR_ITERATOR_H
#define EXECUTOR_ITERATOR_H

#include <type_traits>
#include "soa/constants.h"

namespace ikra {
namespace executor {

using ikra::soa::IndexType;

// This class implements an iterator for SOA objects, starting a given ID. It
// points to an object with ID i and forwards the pointer to i + 1. This class
// can handle both valid and zero addressing mode.
template<typename T>
class Iterator {
 private:
  using InnerT = typename std::remove_pointer<T>::type;

 public:
  // This iterator should be used on pointers to SOA objects.
  static_assert(std::is_pointer<T>::value, "Expected pointer type.");

  T& operator*() {
    return position_;
  }

  // This iterator can go in both directions.
  template<int A = InnerT::kAddressMode>
  typename std::enable_if<A == soa::kAddressModeZero, Iterator<T>>::type&
  operator+=(IndexType distance) {
    auto next = reinterpret_cast<uintptr_t>(position_) + distance;
    position_ = reinterpret_cast<T>(next);
    return *this;
  }

  template<int A = InnerT::kAddressMode>
  typename std::enable_if<A != soa::kAddressModeZero, Iterator<T>>::type&
  operator+=(IndexType distance) {
    auto next = reinterpret_cast<uintptr_t>(position_) + A*distance;
    position_ = reinterpret_cast<T>(next);
    return *this;
  }

  template<int A = InnerT::kAddressMode>
  typename std::enable_if<A == soa::kAddressModeZero, Iterator<T>>::type&
  operator-=(IndexType distance) {
    auto next = reinterpret_cast<uintptr_t>(position_) - distance;
    position_ = reinterpret_cast<T>(next);
    return *this;
  }

  template<int A = InnerT::kAddressMode>
  typename std::enable_if<A != soa::kAddressModeZero, Iterator<T>>::type&
  operator-=(IndexType distance) {
    auto next = reinterpret_cast<uintptr_t>(position_) - A*distance;
    position_ = reinterpret_cast<T>(next);
    return *this;
  }

  Iterator<T>& operator++() {
    return this->operator+=(1);
  }

  Iterator<T>& operator--() {
    return this->operator-=(1);
  }

  bool operator!=(Iterator<T> other) const {
    return position_ != other.position_;
  }

  Iterator(T position) : position_(position) {}

 private:
  T position_;
};

// Helper function/constructor for Iterator for automatic template parameter
// deduction.
template<typename T>
Iterator<T> make_iterator(T position) {
  return Iterator<T>(position);
}

}  // namespace executor
}  // namespace ikra

#endif  // EXECUTOR_ITERATOR_H