find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

add_executable(particle_simulation particle_simulation.cc)
target_link_libraries(particle_simulation ikra ${SDL2_LIBRARIES})

add_executable(n_body n_body.cc)
target_link_libraries(n_body ikra ${SDL2_LIBRARIES})

add_subdirectory(bfs)

add_subdirectory(traffic)
