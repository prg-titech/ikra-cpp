#include "executor/executor.h"
#include "soa/soa.h"
#include "executor/cuda_executor.h"

static const uint32_t kNumCells = 4624613;
static const uint32_t kNumCars = 110000;
static const uint32_t kNumSharedSignalGroups = 140000;

static const uint32_t kArrayInlineSizeOutgoingCells = 10;  // 4
static const uint32_t kArrayInlineSizeIncomingCells = 10;  // 4
static const uint32_t kArrayInlineSizePath = 10;           // 6
static const uint32_t kArrayInlineSizeSignalGroupCells = 4;

using ikra::soa::SoaLayout;
using ikra::soa::kAddressModeZero;
using ikra::soa::StaticStorageWithArena;

class Car;

class Cell : public SoaLayout<
    Cell, kNumCells, kAddressModeZero,
    StaticStorageWithArena<kNumCells*8*sizeof(uint32_t)>> {
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

  __host__ __device__ Cell(uint32_t max_velocity, double x, double y,
                           uint32_t num_incoming, Cell** incoming,
                           uint32_t num_outgoing, Cell** outgoing,
                           Car* car, bool is_free, bool is_sink,
                           Type type = kResidential)
      : max_velocity_(max_velocity), x_(x), y_(y), type_(type),
        num_incoming_cells_(num_incoming), num_outgoing_cells_(num_outgoing),
        incoming_cells_(num_incoming), outgoing_cells_(num_outgoing),
        car_(car), is_free_(is_free), is_sink_(is_sink) {
    for (uint32_t i = 0; i < num_incoming; ++i) {
      incoming_cells_[i] = incoming[i];
    }

    for (uint32_t i = 0; i < num_outgoing; ++i) {
      outgoing_cells_[i] = outgoing[i];
    }

    controller_max_velocity_ = max_velocity_;
  }

  // Overload: Provide cell indices instead of pointers.
  __host__ __device__ Cell(uint32_t max_velocity, double x, double y,
                           uint32_t num_incoming, unsigned int* incoming,
                           uint32_t num_outgoing, unsigned int* outgoing,
                           Car* car, bool is_free, bool is_sink,
                           Type type = kResidential)
      : max_velocity_(max_velocity), x_(x), y_(y), type_(type),
        num_incoming_cells_(num_incoming), num_outgoing_cells_(num_outgoing),
        incoming_cells_(num_incoming), outgoing_cells_(num_outgoing),
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

  // Sets the maximum temporary speed limit (traffic controller).
  __device__ void set_controller_max_velocity(uint32_t velocity) {
    controller_max_velocity_ = velocity;
  }

  // Removes the maximum temporary speed limit.
  __device__ void remove_controller_max_velocity() {
    controller_max_velocity_ = max_velocity_;
  }

  // Incoming cells.
  array_(Cell*, kArrayInlineSizeIncomingCells, inline_soa) incoming_cells_;
  uint32_t_ num_incoming_cells_;
  __device__ uint32_t num_incoming_cells() const {
    return num_incoming_cells_;
  }
  __device__ Cell* incoming_cell(uint32_t index) const {
    return incoming_cells_[index];
  }

  // Outgoing cells.
  array_(Cell*, kArrayInlineSizeOutgoingCells, inline_soa) outgoing_cells_;
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
  double_ y_;
};

IKRA_DEVICE_STORAGE(Cell);

class Car : public SoaLayout<
    Car, kNumCars, kAddressModeZero,
    StaticStorageWithArena<kNumCars*50*sizeof(uint32_t)>> {
 public:
  IKRA_INITIALIZE_CLASS

  __device__ __host__ Car(bool is_active, uint32_t velocity,
                          uint32_t max_velocity, uint32_t random_state,
                          Cell* position)
      : is_active_(is_active), velocity_(velocity), path_length_(0),
        path_(max_velocity), random_state_(random_state), position_(position),
        max_velocity_(max_velocity) {
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
  array_(Cell*, kArrayInlineSizePath, inline_soa) path_;
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

class SharedSignalGroup : public SoaLayout<SharedSignalGroup,
    kNumSharedSignalGroups, kAddressModeZero,
    StaticStorageWithArena<kNumSharedSignalGroups*4*sizeof(uint32_t)>> {
 public:
  IKRA_INITIALIZE_CLASS
  
  __device__ SharedSignalGroup(uint32_t num_cells, unsigned int* cells)
      : num_cells_(num_cells), cells_(num_cells) {
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

  array_(Cell*, kArrayInlineSizeSignalGroupCells, inline_soa) cells_;
  uint32_t_ num_cells_;
  __device__ uint32_t num_cells() const { return num_cells_; }
  __device__ Cell* cell(uint32_t index) { return cells_[index]; }
};

IKRA_DEVICE_STORAGE(SharedSignalGroup);

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
    // TODO: Add check here to see if cell is free.
    cell = path_[i];
  }

  position()->release();
  cell->occupy(this);
  position_ = cell;

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
  // TODO
}

#include "simulation_adapter.cuh"
#include "traffic-simulation/traffic_aos_int_cuda.h"

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
    printf("velocity[%i] = %i\n", i, counter[i]);
  }
  printf("Inactive cars: %i\n", inactive);
}

void run_simulation() {
  for (int i = 0; i < 100; ++i) {
    // Now start simulation.
    cuda_execute(&Car::step_prepare_path);
    cuda_execute(&Car::step_move);
    cuda_execute(&Car::step_reactivate);
    gpuErrchk(cudaDeviceSynchronize());
  }
}

#include <iostream>
#include <chrono>

template<typename TimeT = std::chrono::milliseconds>
struct measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep execution(F&& func, Args&&... args)
    {
        auto start = std::chrono::steady_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast< TimeT> 
                            (std::chrono::steady_clock::now() - start);
        return duration.count();
    }
};

int main(int argc, char** argv) {
  Cell::initialize_storage();
  Car::initialize_storage();

  load_simulation(argc, argv);

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

  uint64_t time_action = measure<>::execution(run_simulation);
  printf("Time for simulation: %lu\n", time_action);

  print_velocity_histogram<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());
}
