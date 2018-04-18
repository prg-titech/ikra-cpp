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
