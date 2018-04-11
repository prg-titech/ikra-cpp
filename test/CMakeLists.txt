add_executable(array_test soa/array_test.cc)
target_link_libraries(array_test gtest gtest_main ikra)
add_test(array_test ${CMAKE_BINARY_DIR}/bin/array_test)

add_executable(basic_class_test soa/basic_class_test.cc)
target_link_libraries(basic_class_test gtest gtest_main ikra)
add_test(basic_class_test ${CMAKE_BINARY_DIR}/bin/basic_class_test)

add_executable(pointer_arithmetics_test soa/pointer_arithmetics_test.cc)
target_link_libraries(pointer_arithmetics_test gtest gtest_main ikra)
add_test(pointer_arithmetics_test ${CMAKE_BINARY_DIR}/bin/pointer_arithmetics_test)

add_executable(executor_test executor_test.cc)
target_link_libraries(executor_test gtest gtest_main ikra)
add_test(executor_test ${CMAKE_BINARY_DIR}/bin/executor_test)


# CUDA Tests
# TODO: cuda_add_library does not support INTERFACE libraries
include_directories(${CMAKE_SOURCE_DIR}/ikra ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
cuda_add_executable(minimum_cuda_test minimum_cuda_test.cu)
target_link_libraries(minimum_cuda_test gtest gtest_main ikra)
add_test(minimum_cuda_test ${CMAKE_BINARY_DIR}/bin/minimum_cuda_test)

include_directories(${CMAKE_SOURCE_DIR}/ikra ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
cuda_add_executable(array_cuda_test array_cuda_test.cu)
target_link_libraries(array_cuda_test gtest gtest_main ikra)
add_test(array_cuda_test ${CMAKE_BINARY_DIR}/bin/array_cuda_test)

include_directories(${CMAKE_SOURCE_DIR}/ikra ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
cuda_add_executable(cuda_array_test soa/cuda_array_test.cu)
target_link_libraries(cuda_array_test gtest gtest_main ikra)
add_test(cuda_array_test ${CMAKE_BINARY_DIR}/bin/cuda_array_test)

include_directories(${CMAKE_SOURCE_DIR}/ikra ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
cuda_add_executable(cuda_inline_array_memcpy_test soa/cuda_inline_array_memcpy_test.cu)
target_link_libraries(cuda_inline_array_memcpy_test gtest gtest_main ikra)
add_test(cuda_inline_array_memcpy_test ${CMAKE_BINARY_DIR}/bin/cuda_inline_array_memcpy_test)
