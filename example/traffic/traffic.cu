#include <limits>
#include <iostream>

#include "executor/executor.h"
#include "soa/soa.h"
#include "executor/cuda_executor.h"

#include "benchmark.h"
#include "configuration.h"

using ikra::soa::SoaLayout;
using ikra::soa::kAddressModeZero;
using ikra::soa::StaticStorageWithArena;

class Car;

class Cell : public SoaLayout<Cell, kNumCells, kAddressModeZero,
                              StaticStorageWithArena<kCellArenaSize>> {
 public:
  IKRA_INITIALIZE_CLASS

  enum Type {
    // Sorted from smallest to largest.
    kResidential,
    kTertiary,
    kSecondary,
    kPrimary,
    kMotorwayLink,
    kMotorway,

    kMaxType
  };

  // Default constructor, where incoming/outgoing cell lists are passed as
  // Cell pointers.
  __host__ __device__ Cell(uint32_t max_velocity, double x, double y,
                           uint32_t num_incoming, Cell** incoming,
                           uint32_t num_outgoing, Cell** outgoing,
                           Car* car, bool is_free, bool is_sink,
                           uint32_t controller_max_velocity,
                           Type type = kResidential)
      : max_velocity_(max_velocity), x_(x), y_(y), type_(type),
        num_incoming_cells_(num_incoming), num_outgoing_cells_(num_outgoing),
#ifdef ARRAY_CELL_IS_PARTIAL
        incoming_cells_(num_incoming), outgoing_cells_(num_outgoing),
#endif  // ARRAY_CELL_IS_PARTIAL
        car_(car), is_free_(is_free), is_sink_(is_sink),
        controller_max_velocity_(controller_max_velocity) {
    for (uint32_t i = 0; i < num_incoming; ++i) {
      incoming_cells_[i] = incoming[i];
    }

    for (uint32_t i = 0; i < num_outgoing; ++i) {
      outgoing_cells_[i] = outgoing[i];
    }
  }

  // Overload: Provide cell indices instead of pointers.
  __host__ __device__ Cell(uint32_t max_velocity, double x, double y,
                           uint32_t num_incoming, unsigned int* incoming,
                           uint32_t num_outgoing, unsigned int* outgoing,
                           Car* car, bool is_free, bool is_sink,
                           uint32_t controller_max_velocity,
                           Type type = kResidential)
      : max_velocity_(max_velocity), x_(x), y_(y), type_(type),
        num_incoming_cells_(num_incoming), num_outgoing_cells_(num_outgoing),
#ifdef ARRAY_CELL_IS_PARTIAL
        incoming_cells_(num_incoming), outgoing_cells_(num_outgoing),
#endif  // ARRAY_CELL_IS_PARTIAL
        car_(car), is_free_(is_free), is_sink_(is_sink) {
    for (uint32_t i = 0; i < num_incoming; ++i) {
      incoming_cells_[i] = Cell::get_uninitialized(incoming[i]);
    }

    for (uint32_t i = 0; i < num_outgoing; ++i) {
      outgoing_cells_[i] = Cell::get_uninitialized(outgoing[i]);
    }

    controller_max_velocity_ = max_velocity_;
  }

  // A cell is free if is does not contain a car.
  bool_ is_free_;
  __device__ bool is_free() const { return is_free_; }

  // A cell is usually a sink if does not have any outgoing edges.
  bool_ is_sink_;
  __device__ bool is_sink() const { return is_sink_; }

  // Return the maximum velocity that is allowed on this street in general.
  uint32_t_ max_velocity_;

  // Return max. velocity allowed with respect to traffic controllers.
  uint32_t_ controller_max_velocity_;

  // Returns the maximum velocity allowed on this cell at this moment. This
  // function takes into account velocity limitations due to traffic lights.
  __device__ uint32_t max_velocity() const {
    return controller_max_velocity_ < max_velocity_
        ? controller_max_velocity_
        : max_velocity_;
  }

  // The maximum velocity allowed on this cell regardless of
  // traffic controllers.
  __device__ uint32_t street_max_velocity() const {
    return max_velocity_;
  }

  // Sets the maximum temporary speed limit (traffic controller).
  __device__ void set_controller_max_velocity(uint32_t velocity) {
    controller_max_velocity_ = velocity;
  }

  // Removes the maximum temporary speed limit.
  __device__ void remove_controller_max_velocity() {
    controller_max_velocity_ = max_velocity_;
  }

  // Incoming cells.
  ARRAY_CELL_INCOMING incoming_cells_;
  uint32_t_ num_incoming_cells_;
  __device__ uint32_t num_incoming_cells() const {
    return num_incoming_cells_;
  }
  __device__ Cell* incoming_cell(uint32_t index) const {
    return incoming_cells_[index];
  }

  // Outgoing cells.
  ARRAY_CELL_OUTGOING outgoing_cells_;
  uint32_t_ num_outgoing_cells_;
  __device__ uint32_t num_outgoing_cells() const {
    return num_outgoing_cells_;
  }
  __device__ Cell* outgoing_cell(uint32_t index) const {
    return outgoing_cells_[index];
  }

  // The car that is currently occupying this cell (if any).
  field_(Car*) car_;

  // A car enters this cell.
  __device__ void occupy(Car* car) {
    assert((bool) is_free_);
    assert(car_ == nullptr);
    
    car_ = car;
    is_free_ = false;
  }

  // A car leaves this cell.
  __device__ void release() {
    car_ = nullptr;
    is_free_ = true;
  }

  // The type of this cell according to OSM data.
  field_(Type) type_;
  __device__ Type type() const { return type_; }

  // x and y coordinates, only for rendering and debugging purposes.
  double_ x_;
  __device__ double x() const { return x_; }

  double_ y_;
  __device__ double y() const { return y_; }
};

IKRA_DEVICE_STORAGE(Cell);

class Car : public SoaLayout<Car, kNumCars, kAddressModeZero,
                             StaticStorageWithArena<kCarArenaSize>> {
 public:
  IKRA_INITIALIZE_CLASS

  __device__ __host__ Car(bool is_active, uint32_t velocity,
                          uint32_t max_velocity, uint32_t random_state,
                          Cell* position)
      : is_active_(is_active), velocity_(velocity), path_length_(0),
#ifdef ARRAY_CAR_IS_PARTIAL
        path_(max_velocity),
#endif  // ARRAY_CAR_IS_PARTIAL
        max_velocity_(max_velocity), random_state_(random_state),
        position_(position) {
    assert(is_active);
  }

  // If a car enters a sink, it is removed from the simulation (inactive)
  // for a short time.
  bool_ is_active_;
  __device__ bool is_active() const { return is_active_; }

  // The velocity of the car in cells/iteration.
  uint32_t_ velocity_;
  __device__ uint32_t velocity() const { return velocity_; }

  // The max. possible velocity of this car.
  uint32_t_ max_velocity_;
  __device__ uint32_t max_velocity() const { return max_velocity_; }

  // An array of cells that the car will move onto next.
  ARRAY_CAR_PATH path_;
  uint32_t_ path_length_;

  // The current position of the car.
  field_(Cell*) position_;
  __device__ Cell* position() const { return position_; }

  // Every car has a random state to allow for reproducible results.
  uint32_t_ random_state_;

  __device__ uint32_t rand32() {
    // Advance and return random state.
    // Source: https://en.wikipedia.org/wiki/Lehmer_random_number_generator
    random_state_ = static_cast<uint32_t>(
        static_cast<uint64_t>(random_state()) * 279470273u) % 0xfffffffb;
    return random_state_;
  }

  __device__ uint32_t rand32(uint32_t max_value) {
    return rand32() % max_value;
  }

  __device__ uint32_t random_state() const {
    return random_state_;
  }

  __device__ void step_prepare_path() {
    if (is_active()) {
      step_initialize_iteration();
      step_accelerate();
      step_extend_path();
      step_constraint_velocity();
      step_slow_down();
    }
  }

  __device__ Cell* next_step(Cell* cell);

  __device__ void step_initialize_iteration();

  __device__ void step_accelerate();

  __device__ void step_extend_path();

  __device__ void step_constraint_velocity();

  __device__ void step_move();

  __device__ void step_reactivate();

  __device__ void step_slow_down() {
    if (rand32(1000) < 200 && velocity_ > 0) {
      velocity_ = velocity_ - 1;
    }
  }
};

IKRA_DEVICE_STORAGE(Car);

class SharedSignalGroup : public SoaLayout<
    SharedSignalGroup, kNumSharedSignalGroups, kAddressModeZero,
    StaticStorageWithArena<kSharedSignalGroupArenaSize>> {
 public:
  IKRA_INITIALIZE_CLASS
  
  __device__ SharedSignalGroup(uint32_t num_cells, unsigned int* cells) :
#ifdef ARRAY_SIGNAL_GROUP_IS_PARTIAL 
      cells_(num_cells),
#endif  // ARRAY_SIGNAL_GROUP_IS_PARTIAL
      num_cells_(num_cells) {
    for (uint32_t i = 0; i < num_cells; ++i) {
      cells_[i] = Cell::get_uninitialized(cells[i]);
    }
  }

  __device__ void signal_go() {
    for (uint32_t i = 0; i < num_cells_; ++i) {
      cells_[i]->remove_controller_max_velocity();
    }
  }

  __device__ void signal_stop() {
    for (uint32_t i = 0; i < num_cells_; ++i) {
      cells_[i]->set_controller_max_velocity(0);
    }
  }

  ARRAY_SIGNAL_GROUP_CELLS cells_;
  uint32_t_ num_cells_;
  __device__ uint32_t num_cells() const { return num_cells_; }
  __device__ Cell* cell(uint32_t index) { return cells_[index]; }
};

IKRA_DEVICE_STORAGE(SharedSignalGroup);

class TrafficLight : public SoaLayout<
    TrafficLight, kNumTrafficLights, kAddressModeZero, 
    StaticStorageWithArena<kTrafficLightArenaSize>> {
 public:
   IKRA_INITIALIZE_CLASS

  __device__ TrafficLight(uint32_t timer, uint32_t phase_time, uint32_t phase,
                          unsigned int* signal_groups,
                          uint32_t num_signal_groups)
      : timer_(timer), phase_time_(phase_time), phase_(phase),
#ifdef ARRAY_TRAFFIC_LIGHT_IS_PARTIAL
        signal_groups_(num_signal_groups),
#endif  // ARRAY_TRAFFIC_LIGHT_IS_PARTIAL
        num_signal_groups_(num_signal_groups) {
    for (uint32_t i = 0; i < num_signal_groups; ++i) {
      signal_groups_[i] =
          SharedSignalGroup::get_uninitialized(signal_groups[i]);
    }
  }

  __device__ void initialize() {
    for (uint32_t i = 0; i < num_signal_groups_; ++i) {
      signal_groups_[i]->signal_stop();
    }
  }

  __device__ void step() {
    timer_ = (timer_ + 1) % phase_time_;

    if (timer_ == 0) {
      signal_groups_[phase_]->signal_stop();
      phase_ = (phase_ + 1) % num_signal_groups_;
      signal_groups_[phase_]->signal_go();
    }
  }

  // This timer is increased with every step.
  uint32_t_ timer_;

  uint32_t_ phase_time_;

  uint32_t_ phase_;

  ARRAY_TRAFFIC_LIGHT_SIGNAL_GROUPS signal_groups_;
  uint32_t_ num_signal_groups_;
};

IKRA_DEVICE_STORAGE(TrafficLight);

class PriorityYieldTrafficController : public SoaLayout<
    PriorityYieldTrafficController, kNumPriorityCtrls, kAddressModeZero,
    StaticStorageWithArena<kPriorityCtrlArenaSize>> {
 public:
  IKRA_INITIALIZE_CLASS

  __device__ PriorityYieldTrafficController(unsigned int* signal_groups,
                                            uint32_t num_signal_groups) :
#ifdef ARRAY_PRIORITY_CTRL_IS_PARTIAL 
        signal_groups_(num_signal_groups),
#endif  // ARRAY_PRIORITY_CTRL_IS_PARTIAL
        num_signal_groups_(num_signal_groups) {
    for (uint32_t i = 0; i < num_signal_groups; ++i) {
      signal_groups_[i] =
          SharedSignalGroup::get_uninitialized(signal_groups[i]);
    }
  }

  __device__ void initialize() {
    for (uint32_t i = 0; i < num_signal_groups_; ++i) {
      signal_groups_[i]->signal_stop();
    }
  }

  __device__ void step() {
    bool found_traffic = false;
    // Cells are sorted by priority.
    for (uint32_t i = 0; i < num_signal_groups_; ++i) {
      SharedSignalGroup* next_group = signal_groups_[i];
      bool has_incoming = has_incoming_traffic(next_group);
      uint32_t num_cells = next_group->num_cells();

      if (!found_traffic && has_incoming) {
        found_traffic = true;
        // Allow traffic to flow.
        for (uint32_t j = 0; j < num_cells; ++j) {
          next_group->cell(j)->remove_controller_max_velocity();
        }
      } else if (has_incoming) {
        // Traffic with higher priority is incoming.
        for (uint32_t j = 0; j < num_cells; ++j) {
          next_group->cell(j)->set_controller_max_velocity(0);
        }
      }
    }
  }

  ARRAY_PRIORITY_CTRL_SIGNAL_GROUPS signal_groups_;
  uint32_t_ num_signal_groups_;

  __device__ bool has_incoming_traffic(SharedSignalGroup* group) const {
    const uint32_t num_cells = group->num_cells();
    for (uint32_t i = 0; i < num_cells; ++i) {
      Cell* next_cell = group->cell(i);

      // Report incoming traffic if at least one cells in the group reports
      // incoming traffic.
      if (has_incoming_traffic(next_cell, next_cell->street_max_velocity())) {
        return true;
      }
    }
    return false;
  }

  __device__ bool has_incoming_traffic(Cell* cell, uint32_t lookahead) const {
    if (lookahead == 0) {
      return !cell->is_free();
    }

    // Check incoming cells. This is BFS.
    const uint32_t num_incoming = cell->num_incoming_cells();
    for (uint32_t i = 0; i < num_incoming; ++i) {
      Cell* next_cell = cell->incoming_cell(i);
      if (has_incoming_traffic(next_cell, lookahead - 1)) {
        return true;
      }
    }

    return !cell->is_free();
  }
};

IKRA_DEVICE_STORAGE(PriorityYieldTrafficController)

class Simulation : public SoaLayout<Simulation, 1> {
 public:
  IKRA_INITIALIZE_CLASS

  __device__ Simulation() : random_state_(123) {}

  uint32_t_ random_state_;

  __device__ uint32_t rand32() {
    // Advance and return random state.
    // Source: https://en.wikipedia.org/wiki/Lehmer_random_number_generator
    random_state_ = static_cast<uint32_t>(
        static_cast<uint64_t>(random_state()) * 279470273u) % 0xfffffffb;
    return random_state_;
  }

  __device__ uint32_t rand32(uint32_t max_value) {
    return rand32() % max_value;
  }

  __device__ uint32_t random_state() const {
    return random_state_;
  }

  __device__ void step_random_state() {
    rand32();
  }

  __device__ Cell* random_free_cell(Car* car) {
    uint64_t state = random_state();
    uint64_t num_cars = Car::size();
    uint64_t num_cells = Cell::size();
    uint32_t max_tries = num_cells / num_cars;

    for (uint32_t i = 0; i < max_tries; ++i) {
      uint64_t cell_id = (num_cells * (car->id() + state) / num_cars + i)
                         % num_cells;
      Cell* next_cell = Cell::get(cell_id);
      if (next_cell->is_free()) {
        return next_cell;
      }
    }

    // Could not find free cell. Try again in next iteration.
    return nullptr;
  }
};

IKRA_DEVICE_STORAGE(Simulation);


__device__ Cell* Car::next_step(Cell* position) {
  // Random walk.
  uint32_t num_cells = position->num_outgoing_cells();
  return position->outgoing_cell(rand32(num_cells));
}

__device__ void Car::step_initialize_iteration() {
  // Reset calculated path. This forces cars with a random moving behavior to
  // select a new path in every iteration. Otherwise, cars might get "stucjk"
  // on a full network if many cars are waiting for the one in front of them in
  // a cycle.
  // TODO: Check if we can keep the path at least partially somehow.
  path_length_ = 0;
}

__device__ void Car::step_accelerate() {
  // Speed up the car by 1 or 2 units.
  uint32_t speedup = rand32(2) + 1;
  velocity_ = max_velocity_ < velocity_ + speedup
      ? static_cast<uint32_t>(max_velocity_) : velocity_ + speedup;
}

__device__ void Car::step_extend_path() {
  Cell* cell = position_;

  for (uint32_t i = 0; i < velocity_; ++i) {
    if (cell->is_sink()) {
      break;
    }

    cell = next_step(cell);
    path_[i] = cell;
    path_length_ = path_length_ + 1;
  }

  velocity_ = path_length_;
}

__device__ void Car::step_constraint_velocity() {
  // This is actually only needed for the very first iteration, because a car
  // may be positioned on a traffic light cell.
  if (velocity_ > position()->max_velocity()) {
    velocity_ = position()->max_velocity();
  }

  uint32_t path_index = 0;
  int distance = 1;

  while (distance <= velocity_) {
    // Invariant: Movement of up to `distance - 1` many cells at `velocity_`
    //            is allowed.
    // Now check if next cell can be entered.
    Cell* next_cell = path_[path_index];

    // Avoid collision.
    if (!next_cell->is_free()) {
      // Cannot enter cell.
      --distance;
      velocity_ = distance;
      break;
    } // else: Can enter next cell.

    if (velocity_ > next_cell->max_velocity()) {
      // Car is too fast for this cell.
      if (next_cell->max_velocity() > distance - 1) {
        // Even if we slow down, we would still make progress.
        velocity_ = next_cell->max_velocity();
      } else {
        // Do not enter the next cell.
        --distance;
        velocity_ = distance;
        break;
      }
    }

    ++distance;
    ++path_index;
  }

  --distance;

  // TODO: Check why the cast is necessary.
  assert(distance <= (int) velocity());
}

__device__ void Car::step_move() {
  Cell* cell = position_;
  for (int i = 0; i < velocity_; ++i) {
    cell = path_[i];
    position()->release();
    cell->occupy(this);
    position_ = cell;
  }

  if (position()->is_sink()) {
    // Remove car from the simulation. Will be added again in the next
    // iteration.
    position()->release();
    path_length_ = 0;
    is_active_ = false;
  }
}

__device__ void Car::step_reactivate() {
  if (!is_active()) {
    Cell* free_cell = Simulation::get(0)->random_free_cell(this);

    if (free_cell != nullptr) {
      position_ = free_cell;
      is_active_ = true;
      path_length_ = 0;
      free_cell->occupy(this);
    }
  }
}

#include "simulation_adapter.cuh"
#include "traffic-simulation/traffic_aos_int_cuda.h"
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
  cuda_execute(&Car::step_move);
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
  load_simulation(argc, argv);
  benchmark();

  print_velocity_histogram<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
  caluclate_checksum<<<1,1>>>();
  gpuErrchk(cudaDeviceSynchronize());
}
