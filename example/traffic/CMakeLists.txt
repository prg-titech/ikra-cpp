set(BENCH_CELL_IN_MODE "--" CACHE STRING "")
ADD_DEFINITIONS(-DBENCH_CELL_IN_MODE=${BENCH_CELL_IN_MODE})

set(BENCH_CELL_OUT_MODE "--" CACHE STRING "")
ADD_DEFINITIONS(-DBENCH_CELL_OUT_MODE=${BENCH_CELL_OUT_MODE})

set(BENCH_CAR_MODE "--" CACHE STRING "")
ADD_DEFINITIONS(-DBENCH_CAR_MODE=${BENCH_CAR_MODE})

set(BENCH_SIGNAL_GROUP_MODE "--" CACHE STRING "")
ADD_DEFINITIONS(-DBENCH_SIGNAL_GROUP_MODE=${BENCH_SIGNAL_GROUP_MODE})

set(BENCH_TRAFFIC_LIGHT_MODE "--" CACHE STRING "")
ADD_DEFINITIONS(-DBENCH_TRAFFIC_LIGHT_MODE=${BENCH_TRAFFIC_LIGHT_MODE})

set(BENCH_LAYOUT_MODE "--" CACHE STRING "")
ADD_DEFINITIONS(-DBENCH_LAYOUT_MODE=${BENCH_LAYOUT_MODE})

set(BENCHMARK_MODE "0" CACHE STRING "")
ADD_DEFINITIONS(-DBENCHMARK_MODE=${BENCHMARK_MODE})


# CUDA
# TODO: cuda_add_library does not support INTERFACE libraries
include_directories(${CMAKE_SOURCE_DIR}/ikra)
cuda_add_executable(traffic traffic.cu simulation_adapter.cu traffic-simulation/graphml_network_builder.cc traffic-simulation/traffic.cc traffic-simulation/simple_network_builder.cc traffic-simulation/drawing.cc traffic-simulation/traffic_aos_int_cuda.cu traffic-simulation/traffic_aos_int.cc)
target_link_libraries(traffic ikra SDL2 SDL2_gfx)
