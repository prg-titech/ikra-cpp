#include <tuple>    // TODO: Only need uintptr_t here.
#include <type_traits>

#ifndef EXECUTOR_ARRAY_H
#define EXECUTOR_ARRAY_H

namespace ikra {
namespace executor {

// This function executes a method on all SOA objects that are enumerated by an
// iterator (between begin and end). This is currently a fully sequential
// operation.
template<typename T, typename F, typename... Args>
void execute(T begin, T end, F function, Args... args) {
  for (auto iter = begin; iter != end; ++iter) {
    // TODO: Should we override and use operator->* here?
    ((**iter).*function)(args...);
  }
}

// This function executes a method on all SOA objects that are enumerated by an
// iterator (length many elements). This is currently a fully sequential
// operation.
template<typename T, typename F, typename... Args>
void execute(T begin, uintptr_t length, F function, Args... args) {
  T iter = begin;
  for (uint32_t i = 0; i < length; ++i, ++iter) {
    ((**iter).*function)(args...);
  }
}

// This function executes a method on all SOA objects of a given class.
// This is currently a fully sequential operation.
// TODO: Is is possible to infer T?
template<typename T, typename F, typename... Args>
void execute(F function, Args... args) {
  execute(T::begin(), T::end(), function, args...);
}

}  // namespace executor
}  // namespace ikra

#endif  // EXECUTOR_ARRAY_H
