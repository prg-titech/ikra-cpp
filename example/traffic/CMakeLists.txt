set(num_cars "-1" CACHE STRING "")
ADD_DEFINITIONS(-DSIM_NUM_CARS=${num_cars})


# CUDA
# TODO: cuda_add_library does not support INTERFACE libraries
include_directories(${CMAKE_SOURCE_DIR}/ikra)
cuda_add_executable(traffic traffic.cu simulation_adapter.cu traffic-simulation/graphml_network_builder.cc traffic-simulation/traffic.cc traffic-simulation/simple_network_builder.cc traffic-simulation/drawing.cc traffic-simulation/traffic_aos_int_cuda.cu traffic-simulation/traffic_aos_int.cc)
target_link_libraries(traffic ikra SDL2 SDL2_gfx)
