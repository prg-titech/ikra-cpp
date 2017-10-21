#include <type_traits>

#ifndef EXECUTOR_ARRAY_H
#define EXECUTOR_ARRAY_H

namespace ikra {
namespace executor {

// This class executes a method on all SOA objects that are enumerated by an
// iterator. This is currently a fully sequential operation.
template<typename T>
class IteratorExecutor_ {
 public:
  IteratorExecutor_(T begin, T end)
      : begin_(begin), end_(end) {}

  template<typename F, typename... Args>
  void execute(F function, Args... args) {
    for (auto iter = begin_; iter != end_; ++iter) {
      // TODO: Should we override and use operator->* here?
      ((**iter).*function)(args...);
    }
  }

 private:
  T begin_;
  T end_;
};

// Helper function/constructor for IteratorExecutor_ for automatic template 
// parameter deduction.
template<typename T>
IteratorExecutor_<T> IteratorExecutor(T begin, T end) {
  return IteratorExecutor_<T>(begin, end);
}

}  // namespace executor
}  // namespace ikra

#endif  // EXECUTOR_ARRAY_H
