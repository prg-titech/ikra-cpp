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


// Defined here because Simulation must be defined.
__device__ void Car::step_reactivate() {
  if (!is_active()) {
    Cell* free_cell = Simulation::get(0)->random_free_cell(this);

    if (free_cell != nullptr) {
      assert(free_cell->is_free());
      position_ = free_cell;
      is_active_ = true;
      path_length_ = 0;
      velocity_ = 0;
      free_cell->occupy(this);
    }
  }
}
