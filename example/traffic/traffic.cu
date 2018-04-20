#include <limits>
#include <iostream>

#include "executor/executor.h"
#include "soa/soa.h"
#include "executor/cuda_executor.h"

#include "benchmark.h"
#include "configuration.h"
#include "simulation_adapter.h"
#include "traffic-simulation/traffic_aos_int_cuda.h"

using ikra::soa::SoaLayout;
using ikra::soa::kAddressModeZero;
using ikra::soa::StaticStorageWithArena;

class Car;

// TODO: Figure out how extern __device__ variables work in CUDA, so that we
// can use normal header files instead of textual headers.
#include "entities/cell.inc"
#include "entities/car.inc"
#include "entities/shared_signal_group.inc"
#include "entities/traffic_light.inc"
#include "entities/priority_yield_traffic_controller.inc"
#include "entities/simulation.inc"
#include "simulation_converter.inc"
#include "statistics.inc"


void run_traffic_controllers() {
  cuda_execute(&Simulation::step_random_state);
  cuda_execute(&TrafficLight::step);
  cuda_execute(&PriorityYieldTrafficController::step);
  cudaDeviceSynchronize();
}

void run_cars() {
  cuda_execute(&Car::step_prepare_path);
  cudaDeviceSynchronize();

  cuda_execute(&Car::step_move);
  cudaDeviceSynchronize();

  cuda_execute(&Car::step_reactivate);
  cudaDeviceSynchronize();
}

void benchmark() {
  uint64_t time_controllers[kNumBenchmarkRuns] = {0};
  uint64_t time_cars[kNumBenchmarkRuns] = {0};
  uint64_t time_total[kNumBenchmarkRuns] = {0};

  for (uint32_t r = 0; r < kNumBenchmarkRuns; ++r) {
    Cell::initialize_storage();
    Car::initialize_storage();
    SharedSignalGroup::initialize_storage();
    TrafficLight::initialize_storage();
    PriorityYieldTrafficController::initialize_storage();

    convert_simulation();
    cuda_execute(&TrafficLight::initialize);
    cuda_execute(&PriorityYieldTrafficController::initialize);
    cudaDeviceSynchronize();

    for (uint32_t i = 0; i < kNumIterations; ++i) {
      uint64_t t_ctrl = measure<>::execution(run_traffic_controllers);
      time_controllers[r] += t_ctrl;

      uint64_t t_car = measure<>::execution(run_cars);
      time_cars[r] += t_car;

      time_total[r] = time_controllers[r] + time_cars[r];
    }
    gpuErrchk(cudaPeekAtLastError());
  }

  // Find best run.
  uint64_t best_time = std::numeric_limits<uint64_t>::max();
  uint32_t best_index = -1;
  for (uint32_t r = 0; r < kNumBenchmarkRuns; ++r) {
    if (time_total[r] < best_time) {
      best_time = time_total[r];
      best_index = r;
    }
  }

  // Print best run.
  print_statistics(time_cars[best_index]/1000,
                   time_controllers[best_index]/1000,
                   time_total[best_index]/1000);
}

int main(int argc, char** argv) {
  load_simulation(argc, argv, kNumCars);
  benchmark();

  print_histograms_1<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
  print_histograms_2<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
  print_histograms_3<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
  print_histograms_4<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
  print_histograms_5<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
  caluclate_checksum<<<1,1>>>();
  gpuErrchk(cudaDeviceSynchronize());
}
