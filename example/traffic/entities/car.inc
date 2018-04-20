class Car : public SoaLayout<Car, kNumCars, kAddressModeZero,
                             StaticStorageWithArena<kCarArenaSize>,
                             kCarLayoutMode> {
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

  // The velocity of the car in cells/iteration.
  // [ OFFSET = 0 ]
  uint32_t_ velocity_;

  // The max. possible velocity of this car.
  // [ OFFSET = 4 ]
  uint32_t_ max_velocity_;

  // The lengths of the precalculated path.
  // [ OFFSET = 8 ]
  uint32_t_ path_length_;

  // Every car has a random state to allow for reproducible results.
  // [ OFFSET = 12 ]
  uint32_t_ random_state_;

  // An array of cells that the car will move onto next.
  // [ OFFSET = 16 ]
  ARRAY_CAR_PATH path_;

  // The current position of the car.
  // [ OFFSET = 16 + k*PTR_SIZE ]
  field_(Cell*) position_;

  // If a car enters a sink, it is removed from the simulation (inactive)
  // for a short time.
  // [ OFFSET = 16 + k*PTR_SIZE ]
  bool_ is_active_;

  // Add dummy field to make sure that objects are properly aligned in AOS.
  // [ OFFSET = 17 + k*PTR_SIZE ]
  array_(char, kCarAlignmentPadding, object) dummy_alignment_;

  __device__ uint32_t velocity() const { return velocity_; }

  __device__ uint32_t max_velocity() const { return max_velocity_; }

  __device__ Cell* position() const { return position_; }

  __device__ bool is_active() const { return is_active_; }

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


__device__ Cell* Car::next_step(Cell* position) {
  // Almost random walk.
  const uint32_t num_outgoing = position->num_outgoing_cells();
  assert(num_outgoing > 0);

#ifdef MOVE_ONTO_LARGE_STREETS
  // Collect probabilities of outgoing cells.
  int prob_sum = 0;
  for (int i = 0; i < num_outgoing; ++i) {
    Cell* candidate = position->outgoing_cell(i);

    // More likely to drive onto larger roads.
    int difference = candidate->type() - position->type() + Cell::kMaxType;
    prob_sum += 100*difference;
  }

  // Now choose a path randomly.
  int rand_select = rand32(prob_sum);
  int threshold = 0;
  for (int i = 0; i < num_outgoing; ++i) {
    Cell* candidate = position->outgoing_cell(i);

    // More likely to drive onto larger roads.
    int difference = candidate->type() - position->type() + Cell::kMaxType;
    threshold += 100*difference;

    if (rand_select < threshold) {
      assert(candidate != position);
      return candidate;
    }
  }

  // We should never end up here.
  assert(false);
#endif  // MOVE_ONTO_LARGE_STREETS

  // Need some kind of return statement here.
  return position->outgoing_cell(rand32(num_outgoing));
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
  Cell* next_cell;

  for (uint32_t i = 0; i < velocity_; ++i) {
    if (cell->is_sink()) {
      break;
    }

    next_cell = next_step(cell);
    assert(next_cell != cell);

    cell = next_cell;
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
        assert(distance >= 0);

        velocity_ = distance;
        break;
      }
    }

    ++distance;
    ++path_index;
  }

  --distance;


  for (uint32_t i = 0; i < velocity_; ++i) {
    assert(path_[i]->is_free());
    assert(i == 0 || path_[i - 1] != path_[i]);
  }
  // TODO: Check why the cast is necessary.
  assert(distance <= (int) velocity());
}

__device__ void Car::step_move() {
  if (is_active()) {
    Cell* cell = position_;
    for (uint32_t i = 0; i < velocity_; ++i) {
      assert(path_[i] != cell);

      cell = path_[i];
      assert(cell->is_free());

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
}
