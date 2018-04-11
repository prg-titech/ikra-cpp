#include "executor/executor.h"
#include "soa/soa.h"

static const IndexType kNumCells = 1000;
static const IndexType kArrayInlineSizeOutgoingCells = 4;
static const IndexType kArrayInlineSizeIncomingCells = 4;

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

  // A cell is free if is does not contain a car.
  bool_ is_free_;
  bool is_free() const { return is_free_; }

  // A cell is usually a sink if does not have any outgoing edges.
  bool_ is_sink_;
  bool is_sink() const { return is_sink_; }

  // Return the maximum velocity that is allowed on this street in general.
  int_ max_velocity_;

  // Return max. velocity allowed regardless of traffic controllers.
  int_ street_max_velocity_;

  // Returns the maximum velocity allowed on this cell at this moment. This
  // function takes into account velocity limitations due to traffic lights.
  int max_velocity() const {
    return street_max_velocity_ < max_velocity_
        ? street_max_velocity_
        : max_velocity_;
  }

  // Incoming cells.
  array_(Cell*, kArrayInlineSizeIncomingCells, inline_soa) incoming_cells_;
  int_ num_incoming_cells_;

  // Outgoing cells.
  array_(Cell*, kArrayInlineSizeOutgoingCells, inline_soa) outgoing_cells_;
  int_ num_outgoing_cells_;

  // The car that is currently occupying this cell (if any).
  field_(Car*) car_;

  // A car enters this cell.
  void occupy(Car* car) {
    car_ = car;
    is_free_ = false;
  }

  // A car leaves this cell.
  void release() {
    car_ = nullptr;
    is_free_ = true;
  }

  // The type of this cell according to OSM data.
  field_(Type) type_;
  Type type() const { return type_; }

  // x and y coordinates, only for rendering and debugging purposes.
  double_ x_;
  double_ y_;
};

IKRA_DEVICE_STORAGE(Cell);
