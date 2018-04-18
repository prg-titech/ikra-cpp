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
