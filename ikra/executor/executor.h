#ifndef EXECUTOR_EXECUTOR_H
#define EXECUTOR_EXECUTOR_H

// Asserts active only in debug mode (NDEBUG).
#include <cassert>

#include <cstdint>
#include <type_traits>

#include "soa/constants.h"

namespace ikra {
namespace executor {

using ikra::soa::IndexType;

template<class F>
struct FunctionTypeHelper;

template<typename R, typename C, typename... Args>
struct FunctionTypeHelper<R (C::*)(Args...)> {
  using return_type = R;
  using class_type = C;
};

// A convenience class storing two iterators, i.e., a range of SOA objects.
template<typename T>
class IteratorRange {
 public:
  IteratorRange(T begin, T end) : begin_(begin), end_(end) {}

  T begin() {
    return begin_;
  }

  T end() {
    return end_;
  }

 private:
  T begin_;
  T end_;
};

template<typename T, typename... Args>
T* construct(size_t count, Args... args) {
  assert(count > 0);
  T* first = new T(args...);
  for (size_t i = 1; i < count; ++i) {
    new T(args...);
  }
  return first;
}

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
// iterator (between begin and end). It also reduces all return values.
template<typename T, typename F, typename... Args, typename G, typename D>
D execute_and_reduce(T begin, T end, F function, G reduce_function,
                     D default_value, Args... args) {
  if (begin == end) {
    return default_value;
  }

  D reduced_value = ((**begin).*function)(args...);
  for (auto iter = ++begin; iter != end; ++iter) {
    // TODO: Should we override and use operator->* here?
    reduced_value = reduce_function(reduced_value,
                                    ((**iter).*function)(args...));
  }
  return reduced_value;
}

// This function executes a method on all SOA objects that are enumerated by an
// iterator (between begin and end). This is currently a fully sequential
// operation.
template<typename T, typename F, typename... Args>
void execute(IteratorRange<T> range, F function, Args... args) {
  for (auto iter = range.begin(); iter != range.end(); ++iter) {
    ((**iter).*function)(args...);
  }
}

// This function executes a method on all SOA objects that are enumerated by an
// iterator (length many elements). This is currently a fully sequential
// operation.
template<typename T, typename F, typename... Args>
void execute(T begin, IndexType length, F function, Args... args) {
  T iter = begin;
  for (uint32_t i = 0; i < length; ++i, ++iter) {
    ((**iter).*function)(args...);
  }
}

// This function executes a method on all SOA objects of a given class.
// This is currently a fully sequential operation.
template<typename F, typename... Args,
         class T = typename FunctionTypeHelper<F>::class_type>
void execute(F function, Args... args) {
  IndexType num_instances = T::size();
  for (IndexType i = 0; i < num_instances; ++i) {
    ((*T::get(i)).*function)(args...);
  }
}

// This function executes a method on all SOA objects of a given class,
// starting from ID 0 up to a given ID (template parameter).
// This is currently a fully sequential operation.
template<IndexType MaxId, typename F, typename... Args,
         class T = typename FunctionTypeHelper<F>::class_type>
void execute(F function, Args... args) {
  for (IndexType i = 0; i < MaxId; ++i) {
    ((*T::get(i)).*function)(args...);
  }
}

// This function executes a method on all SOA objects of a given class and
// reduces the return values.
template<typename F, typename... Args, typename G, typename D,
         class T = typename FunctionTypeHelper<F>::class_type>
D execute_and_reduce(F function, G reduce_function, D default_value,
                     Args... args) {
  return execute_and_reduce(T::begin(), T::end(),
                            function, reduce_function, default_value, args...);
}

}  // namespace executor
}  // namespace ikra

#endif  // EXECUTOR_EXECUTOR_H
