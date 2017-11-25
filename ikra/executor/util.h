#ifndef EXECUTOR_UTIL_H
#define EXECUTOR_UTIL_H

namespace ikra {
namespace executor {

template<class F>
struct FunctionTypeHelper;

template<typename R, typename C, typename... Args>
struct FunctionTypeHelper<R (C::*)(Args...)> {
  using return_type = R;
  using class_type = C;
};

}  // namespace executor
}  // namespace ikra

#endif  // EXECUTOR_UTIL_H
