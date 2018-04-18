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

__global__ void print_velocity_histogram() {
  int counter[50];
  int inactive = 0;

  for (int i = 0; i < 50; ++i) {
    counter[i] = 0;
  }

  for (int i = 0; i < Car::size(); ++i) {
    counter[Car::get(i)->velocity()]++;
    if (!Car::get(i)->is_active()) {
      inactive++;
    }
  }

  for (int i = 0; i < 50; ++i) {
    if (counter[i] > 0) {
      printf("velocity[%i] = %i\n", i, counter[i]);
    }
  }
  printf("Inactive cars: %i\n", inactive);
}

__global__ void caluclate_checksum() {
  uint64_t c = 17;
  for (uint32_t i = 0; i < Car::size(); ++i) {
    Cell* position = Car::get(i)->position();
    c += position->x() + position->y();
    c %= UINT64_MAX;
  }
  int result = c % 1234567;

  printf("Checksum: %i\n", result);
}

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

void print_statistics(unsigned long time_cars, unsigned long time_controllers,
                      unsigned long time_total) {
  unsigned long utilization_Cell = Cell::storage().arena_utilization();
  unsigned long utilization_Car = Car::storage().arena_utilization();
  unsigned long utilization_SharedSignalGroup =
      SharedSignalGroup::storage().arena_utilization();
  unsigned long utilization_TrafficLight =
      TrafficLight::storage().arena_utilization();
  unsigned long utilization_PriorityCtrl = 
      PriorityYieldTrafficController::storage().arena_utilization();
  unsigned long total_utilization =
      utilization_Cell + utilization_Car + utilization_SharedSignalGroup
      + utilization_TrafficLight + utilization_PriorityCtrl;

  unsigned long long int storage_Cell = Cell::size()*Cell::ObjectSize::value;
  unsigned long long int storage_Car = Car::size()*Car::ObjectSize::value;
  unsigned long long int storage_SignalGroup =
      SharedSignalGroup::size()*SharedSignalGroup::ObjectSize::value;
  unsigned long long int storage_TrafficLight =
      TrafficLight::size()*TrafficLight::ObjectSize::value;
  unsigned long long int storage_PriorityCtrl =
     PriorityYieldTrafficController::size()
          *PriorityYieldTrafficController::ObjectSize::value;

  unsigned long long int regular_storage_size =
      storage_Cell + storage_Car + storage_SignalGroup
      + storage_TrafficLight + storage_PriorityCtrl;

  fprintf(stderr,
      "%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%llu,%llu,%llu,%llu,%llu,%llu\n",
      time_cars,
      time_controllers,
      time_total,
      utilization_Cell,
      utilization_Car,
      utilization_SharedSignalGroup,
      utilization_TrafficLight,
      utilization_PriorityCtrl,
      total_utilization,
      storage_Cell,
      storage_Car,
      storage_SignalGroup,
      storage_TrafficLight,
      storage_PriorityCtrl,
      regular_storage_size);
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

  print_velocity_histogram<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
  caluclate_checksum<<<1,1>>>();
  gpuErrchk(cudaDeviceSynchronize());
}
