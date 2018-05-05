#ifndef EXECUTOR_CUDA_EXECUTOR_MACROS_H
#define EXECUTOR_CUDA_EXECUTOR_MACROS_H

// Templatize function name by current (outer) virtual warp size. Add a dummy
// argument in case no arguments were passed to the function call.
#define IKRA_func_to_vw(func, ...) \
    func<IKRA_extract_virtual_warp_size(__VA_ARGS__, 1)>

#define cuda_execute(func, ...) \
  ikra::executor::cuda::ExecuteKernelProxy<decltype(func), func> \
      ::call<Ikra_VW_SZ>(__VA_ARGS__)

// If running on device, utilize nested parallelism.
#define cuda_execute_vw(func, ...) \
  ikra::executor::cuda::ExecuteKernelProxy< \
      decltype(IKRA_func_to_vw(func, __VA_ARGS__)), \
      IKRA_func_to_vw(func, __VA_ARGS__)> \
          ::call<Ikra_VW_SZ>(__VA_ARGS__)

#define cuda_execute_fixed_size(func, size, ...) \
  ikra::executor::cuda::ExecuteKernelProxy<decltype(func), func> \
      ::call_fixed_size<Ikra_VW_SZ, size>(__VA_ARGS__)

#define cuda_execute_fixed_size_vw(func, size, ...) \
  ikra::executor::cuda::ExecuteKernelProxy< \
      decltype(IKRA_func_to_vw(func, __VA_ARGS__)), \
      IKRA_func_to_vw(func, __VA_ARGS__)> \
          ::call_fixed_size<Ikra_VW_SZ, size>(__VA_ARGS__)

#define cuda_execute_and_return(func, ...) \
  ikra::executor::cuda::ExecuteAndReturnKernelProxy<decltype(func), func> \
      ::call<Ikra_VW_SZ>(__VA_ARGS__)

#define cuda_execute_and_return_vw(func, ...) \
  ikra::executor::cuda::ExecuteAndReturnKernelProxy< \
      decltype(IKRA_func_to_vw(func, __VA_ARGS__)), \
      IKRA_func_to_vw(func, __VA_ARGS__)> \
          ::call<Ikra_VW_SZ>(__VA_ARGS__)

#define cuda_execute_and_reduce(func, ...) \
  ikra::executor::cuda::ExecuteAndReduceKernelProxy<decltype(func), func> \
      ::call<Ikra_VW_SZ>(__VA_ARGS__)

#define cuda_execute_and_reduce_vw(func, ...) \
  ikra::executor::cuda::ExecuteAndReduceKernelProxy< \
      decltype(IKRA_func_to_vw(func, __VA_ARGS__)), \
      IKRA_func_to_vw(func, __VA_ARGS__)> \
          ::call<Ikra_VW_SZ>(__VA_ARGS__)

#endif  // EXECUTOR_CUDA_EXECUTOR_MACROS_H
