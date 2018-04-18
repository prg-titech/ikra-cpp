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
