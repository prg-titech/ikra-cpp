#ifndef EXECUTOR_ITERATOR_H
#define EXECUTOR_ITERATOR_H

#include <type_traits>

namespace ikra {
namespace executor {

template<typename T>
class Iterator_ {
 private:
  using InnerT = typename std::remove_pointer<T>::type;

 public:
  // This iterator should be used on pointers to SOA objects.
  //static_assert(std::is_pointer<T>::value, "Expected pointer type.");
  static_assert(std::is_pointer<T>::value, "Expected pointer type.");

  T& operator*() {
    return position_;
  }

  Iterator_& operator++() {
    auto next = reinterpret_cast<uintptr_t>(position_) + InnerT::kAddressMode;
    position_ = reinterpret_cast<T>(next);
    return *this;
  }

  bool operator!=(Iterator_<T> other) const {
    return position_ != other.position_;
  }

  Iterator_(T position) : position_(position) {}

 private:
  T position_;
};

template<typename T>
Iterator_<T> Iterator(T position) {
  return Iterator_<T>(position);
}

}  // namespace executor
}  // namespace ikra

#endif  // EXECUTOR_ITERATOR_H