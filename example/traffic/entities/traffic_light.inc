
class TrafficLight : public SoaLayout<
    TrafficLight, kNumTrafficLights, kAddressModeZero, 
    StaticStorageWithArena<kTrafficLightArenaSize>,
    kTrafficLightLayoutMode> {
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

  __device__ void step() {
#ifdef SMART_TRAFFIC_LIGHT
  // This is a smart traffic light. If there are cars coming from only one
  // street, give them a green light.
  uint32_t num_found_traffic = 0;
  uint32_t index_last_found = 0;
  for (uint32_t i = 0; i < num_signal_groups_; ++i) {
    SharedSignalGroup* next_group = signal_groups_[i];
    bool has_incoming = has_incoming_traffic(next_group);

    if (has_incoming) {
      num_found_traffic++;
      index_last_found = i;
    }

    if (num_found_traffic > 1) break;
  }

  if (num_found_traffic == 1) {
    // Set this one to green.
    for (uint32_t i = 0; i < num_signal_groups_; ++i) {
      if (i == index_last_found) {
        signal_groups_[i]->signal_go();
      } else {
        signal_groups_[i]->signal_stop();
      }
    }
    timer_ = 0;
    phase_ = index_last_found;
  } else {
    timer_ = (timer_ + 1) % phase_time_;

    if (timer_ == 0) {
      signal_groups_[phase_]->signal_stop();
      phase_ = (phase_ + 1) % num_signal_groups_;
      signal_groups_[phase_]->signal_go();
    }
  }
#else
    timer_ = (timer_ + 1) % phase_time_;

    if (timer_ == 0) {
      signal_groups_[phase_]->signal_stop();
      phase_ = (phase_ + 1) % num_signal_groups_;
      signal_groups_[phase_]->signal_go();
    }
#endif  // SMART_TRAFFIC_LIGHT
  }

  // This timer is increased with every step.
  // [ PADDING = 0 ]
  uint32_t_ timer_;

  // The length of one phase, i.e., number of iterations until the signal
  // changes.
  // [ PADDING = 4 ]
  uint32_t_ phase_time_;

  // An index into signal_groups_, marking the group that currently has a
  // green signal.
  // [ PADDING = 8 ]
  uint32_t_ phase_;

  // Number of signal groups.
  // [ PADDING = 12 ]
  uint32_t_ num_signal_groups_;
  __device__ uint32_t num_signal_groups() const { return num_signal_groups_; }

  // Signal groups controlled by this traffic lights.
  // [ PADDING = 16 ]
  ARRAY_TRAFFIC_LIGHT_SIGNAL_GROUPS signal_groups_;

  // No additional padding required for AOS.
};

IKRA_DEVICE_STORAGE(TrafficLight);
