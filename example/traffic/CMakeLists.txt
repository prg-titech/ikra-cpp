# CUDA
# TODO: cuda_add_library does not support INTERFACE libraries
include_directories(${CMAKE_SOURCE_DIR}/ikra)
cuda_add_executable(traffic traffic.cu)
target_link_libraries(traffic ikra)
