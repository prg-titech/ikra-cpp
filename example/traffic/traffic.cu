#include "executor/executor.h"
#include "soa/soa.h"

static const uint32_t kNumCells = 1000;
static const uint32_t kNumCars = 20;

static const uint32_t kArrayInlineSizeOutgoingCells = 4;
static const uint32_t kArrayInlineSizeIncomingCells = 4;
static const uint32_t kArrayInlineSizePath = 6;

using ikra::soa::SoaLayout;

class Car;

class Cell : public SoaLayout<Cell, kNumCells> {
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
                           Type type = kResidential)
      : max_velocity_(max_velocity), x_(x), y_(y), type_(type),
        num_incoming_cells_(num_incoming), num_outgoing_cells_(num_outgoing),
        incoming_cells_(num_incoming), outgoing_cells_(num_outgoing) {
    for (uint32_t i = 0; i < num_incoming; ++i) {
      incoming_cells_[i] = incoming[i];
    }

    for (uint32_t i = 0; i < num_outgoing; ++i) {
      outgoing_cells_[i] = outgoing[i];
    }
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

class Car : public SoaLayout<Car, kNumCars> {
 public:
  IKRA_INITIALIZE_CLASS

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
    step_initialize_iteration();
    step_accelerate();
    step_extend_path();
    step_constraint_velocity();
  }

  __device__ Cell* next_step(Cell* cell);

  __device__ void step_initialize_iteration();

  __device__ void step_accelerate();

  __device__ void step_extend_path();

  __device__ void step_constraint_velocity();

  __device__ void step_move();

  __device__ void step_reactivate();
};

IKRA_DEVICE_STORAGE(Car);


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
  // TODO: Why does the implicit type cast not work here?
  if (velocity_ > position()->max_velocity()) {
    velocity_ = position()->max_velocity();
  }

  uint32_t distance = 0;
  while (distance < velocity_) {
    // Invariant: Movement of up to `distance` many cells at `velocity_`
    //            is allowed.
    // Now check if next cell can be entered.
    Cell* next_cell = path_[distance];

    // Avoid collision.
    if (!next_cell->is_free()) {
      // Cannot enter cell.
      velocity_ = distance;
      --distance;
      break;
    } // else: Can enter next cell.

    if (velocity_ > next_cell->max_velocity()) {
      // Car is too fast for this cell.
      if (next_cell->max_velocity() > distance) {
        // Even if we slow down, we would still make progress.
        velocity_ = next_cell->max_velocity();
      } else {
        // Do not enter the next cell.
        velocity_ = distance;
        --distance;
        break;
      }
    }

    ++distance;
  }

  --distance;
  assert(distance < velocity_);
}

__device__ void Car::step_move() {
  Cell* cell;
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
  // TODO
}

int main() {
  Cell::initialize_storage();
  Car::initialize_storage();
}
