#include <type_traits>

#ifndef EXECUTOR_ARRAY_H
#define EXECUTOR_ARRAY_H

namespace ikra {
namespace executor {

// This class works only for SOA types.
template<typename T>
class IteratorExecutor_ {
 public:
  IteratorExecutor_(T begin, T end)
      : begin_(begin), end_(end) {}

  template<typename F, typename... Args>
  void execute(F function, Args... args) {
    for (auto iter = begin_; iter != end_; ++iter) {
      ((**iter).*function)(args...);
    }
  }

 private:
  T begin_;
  T end_;
};

template<typename T>
IteratorExecutor_<T> IteratorExecutor(T begin, T end) {
  return IteratorExecutor_<T>(begin, end);
}

}  // namespace executor
}  // namespace ikra

#endif  // EXECUTOR_ARRAY_H
