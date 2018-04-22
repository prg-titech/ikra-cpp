add_executable(bfs_const_size_array bfs_const_size_array.cc)
target_link_libraries(bfs_const_size_array ikra)

add_executable(bfs_inlined_dynamic_array bfs_inlined_dynamic_array.cc)
target_link_libraries(bfs_inlined_dynamic_array ikra)

add_executable(bfs_std_vector bfs_std_vector.cc)
target_link_libraries(bfs_std_vector ikra)


# CUDA
# TODO: cuda_add_library does not support INTERFACE libraries
include_directories(${CMAKE_SOURCE_DIR}/ikra)
cuda_add_executable(bfs_const_size_array_cuda bfs_const_size_array.cu)
target_link_libraries(bfs_const_size_array_cuda ikra)

include_directories(${CMAKE_SOURCE_DIR}/ikra)
cuda_add_executable(bfs_bench bfs_bench.cu)
target_link_libraries(bfs_bench ikra)
