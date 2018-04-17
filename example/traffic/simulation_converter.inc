using IndexType = unsigned int;

// Data storage.
namespace simulation {
namespace aos_int {
extern IndexType s_size_Cell;
extern IndexType s_size_outgoing_cells;
extern IndexType s_size_incoming_cells;
extern IndexType s_size_Car;
extern IndexType s_size_car_paths;
extern IndexType s_size_inactive_cars;
extern IndexType s_size_TrafficLight;
extern IndexType s_size_PriorityYieldTrafficController;
extern IndexType s_size_SharedSignalGroup;
extern IndexType s_size_traffic_light_signal_groups;
extern IndexType s_size_priority_ctrl_signal_groups;
extern IndexType s_size_signal_group_cells;
}  // namespace aos_int

namespace aos_int_cuda {
extern Cell* dev_Cell;
extern IndexType* dev_outgoing_cells;
extern IndexType* dev_incoming_cells;
extern Car* dev_Car;
extern IndexType* dev_car_paths;
extern IndexType* dev_inactive_cars;
extern TrafficLight* dev_TrafficLight;
extern PriorityYieldTrafficController* dev_PriorityYieldTrafficController;
extern SharedSignalGroup* dev_SharedSignalGroup;
extern IndexType* dev_traffic_light_signal_groups;
extern IndexType* dev_priority_ctrl_signal_groups;
extern IndexType* dev_signal_group_cells;
}  // namespace aos_int_cuda
}  // namespace simulation

__global__ void convert_to_ikra_cpp_cells(
    IndexType s_size_Cell,
    simulation::aos_int_cuda::Cell* s_Cell,
    IndexType s_size_outgoing_cells,
    IndexType* s_outgoing_cells,
    IndexType s_size_incoming_cells,
    IndexType* s_incoming_cells) {
  unsigned int tid = blockIdx.x *blockDim.x + threadIdx.x;

  if (tid < s_size_Cell) {
    simulation::aos_int_cuda::Cell& cell = s_Cell[tid];
    Car* car_ptr = cell.car_ == 4294967295
      ? nullptr : Car::get_uninitialized(cell.car_);

    Cell* new_cell = new(Cell::get_uninitialized(tid)) Cell(
        cell.max_velocity_, cell.x_, cell.y_,
        cell.num_incoming_cells_,
        s_incoming_cells + cell.first_incoming_cell_idx_,
        cell.num_outgoing_cells_,
        s_outgoing_cells + cell.first_outgoing_cell_idx_,
        car_ptr, cell.is_free_, cell.is_sink_,
        cell.controller_max_velocity_,
        (Cell::Type) cell.type_);
    assert(new_cell->id() == tid);
  }

  if (tid == 0) {
    Cell::storage().increase_size(s_size_Cell);
  }
}

__global__ void convert_to_ikra_cpp_cars(
    IndexType s_size_Car,
    simulation::aos_int_cuda::Car* s_Car,
    IndexType s_size_Cell) {
  unsigned int tid = blockIdx.x *blockDim.x + threadIdx.x;

  if (tid < s_size_Car) {
    simulation::aos_int_cuda::Car& car = s_Car[tid];

    // Every car must have a position.
    assert(car.position_ != 4294967295);
    assert(car.position_ < s_size_Cell);

    Cell* cell_ptr = car.position_ == 4294967295
      ? nullptr : Cell::get_uninitialized(car.position_);

    Car* new_car = new(Car::get_uninitialized(tid)) Car(
        car.is_active_, car.velocity_, car.max_velocity_,
        car.random_state_, cell_ptr);
    assert(new_car->id() == tid);
  }

  if (tid == 0) {
    Car::storage().increase_size(s_size_Car);
    new Simulation();
  }
}

__global__ void convert_to_ikra_cpp_signal_groups(
    IndexType s_size_SharedSignalGroup,
    simulation::aos_int_cuda::SharedSignalGroup* s_SharedSignalGroup,
    IndexType* s_cells) {
  unsigned int tid = blockIdx.x *blockDim.x + threadIdx.x;

  if (tid < s_size_SharedSignalGroup) {
    simulation::aos_int_cuda::SharedSignalGroup& group =
        s_SharedSignalGroup[tid];

    SharedSignalGroup* new_group =
        new(SharedSignalGroup::get_uninitialized(tid)) SharedSignalGroup(
            group.num_cells_, s_cells + group.first_cell_idx_);
    assert(new_group->id() == tid);
  }

  if (tid == 0) {
    SharedSignalGroup::storage().increase_size(s_size_SharedSignalGroup);
  }
}

__global__ void convert_to_ikra_cpp_traffic_lights(
    IndexType s_size_TrafficLight,
    simulation::aos_int_cuda::TrafficLight* s_TrafficLight,
    IndexType* s_signal_groups) {
  unsigned int tid = blockIdx.x *blockDim.x + threadIdx.x;

  if (tid < s_size_TrafficLight) {
    simulation::aos_int_cuda::TrafficLight& light = s_TrafficLight[tid];

    TrafficLight* new_light = new(TrafficLight::get_uninitialized(tid))
        TrafficLight(light.timer_, light.phase_time_, light.phase_,
                     s_signal_groups + light.first_signal_group_idx_,
                     light.num_signal_groups_);
    assert(new_light->id() == tid);

    new_light->initialize();
  }

  if (tid == 0) {
    TrafficLight::storage().increase_size(s_size_TrafficLight);
  }

}

__global__ void convert_to_ikra_cpp_priority_ctrls(
    IndexType s_size_PriorityYieldTrafficController,
    simulation::aos_int_cuda::PriorityYieldTrafficController*
        s_PriorityYieldTrafficController,
    IndexType* s_signal_groups) {
  unsigned int tid = blockIdx.x *blockDim.x + threadIdx.x;

  if (tid < s_size_PriorityYieldTrafficController) {
    simulation::aos_int_cuda::PriorityYieldTrafficController& ctrl =
        s_PriorityYieldTrafficController[tid];

    PriorityYieldTrafficController* new_ctrl =
        new(PriorityYieldTrafficController::get_uninitialized(tid))
        PriorityYieldTrafficController(
            s_signal_groups + ctrl.first_group_idx_,
            ctrl.num_groups_);
    assert(new_ctrl->id() == tid);

    new_ctrl->initialize();
  }

  if (tid == 0) {
    PriorityYieldTrafficController::storage().increase_size(
        s_size_PriorityYieldTrafficController);
  }
}

void convert_simulation() {
  assert(simulation::aos_int::s_size_Car <= kNumCars);
  assert(simulation::aos_int::s_size_Cell <= kNumCells);

  convert_to_ikra_cpp_cells<<<kNumCells/1024 + 1, 1024>>>(
      simulation::aos_int::s_size_Cell,
      simulation::aos_int_cuda::dev_Cell,
      simulation::aos_int::s_size_outgoing_cells,
      simulation::aos_int_cuda::dev_outgoing_cells,
      simulation::aos_int::s_size_incoming_cells,
      simulation::aos_int_cuda::dev_incoming_cells);
  gpuErrchk(cudaDeviceSynchronize());

  convert_to_ikra_cpp_cars<<<kNumCars/1024 + 1, 1024>>>(
      simulation::aos_int::s_size_Car,
      simulation::aos_int_cuda::dev_Car,
      simulation::aos_int::s_size_Cell);
  gpuErrchk(cudaDeviceSynchronize());

  convert_to_ikra_cpp_signal_groups<<<kNumSharedSignalGroups/1024 + 1, 1024>>>(
      simulation::aos_int::s_size_SharedSignalGroup,
      simulation::aos_int_cuda::dev_SharedSignalGroup,
      simulation::aos_int_cuda::dev_signal_group_cells);
  gpuErrchk(cudaDeviceSynchronize());

  convert_to_ikra_cpp_traffic_lights<<<kNumTrafficLights/1024 + 1, 1024>>>(
      simulation::aos_int::s_size_TrafficLight,
      simulation::aos_int_cuda::dev_TrafficLight,
      simulation::aos_int_cuda::dev_traffic_light_signal_groups);
  gpuErrchk(cudaDeviceSynchronize());

  convert_to_ikra_cpp_priority_ctrls<<<kNumPriorityCtrls/1024 + 1, 1024>>>(
    simulation::aos_int::s_size_PriorityYieldTrafficController,
    simulation::aos_int_cuda::dev_PriorityYieldTrafficController,
    simulation::aos_int_cuda::dev_priority_ctrl_signal_groups);
  gpuErrchk(cudaDeviceSynchronize());
}
