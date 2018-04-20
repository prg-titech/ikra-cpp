#define STRINGIFY(val) #val

#define PRINT_HISTORAM(class, function, maxval) \
{ \
  printf("------------------------------------------------\n"); \
  int counter[maxval];\
  uint64_t sum = 0; \
  uint64_t squared_sum = 0; \
  for (int i = 0; i < maxval; ++i) { \
    counter[i] = 0; \
  } \
  for (uint32_t i = 0; i < class::size(); ++i) { \
    counter[class::get(i)->function()]++; \
    sum += class::get(i)->function(); \
    squared_sum += class::get(i)->function() * class::get(i)->function(); \
  } \
  for (int i = 0; i < maxval; ++i) { \
    if (counter[i] > 0) { \
      printf(STRINGIFY(class) "::" STRINGIFY(function) "[%i] = %i  [ %f % ]\n", \
             i, counter[i], (float) counter[i] * 100.0f / class::size()); \
    } \
  } \
  double mean = sum / (double) class::size(); \
  double variance = (squared_sum - class::size() * mean * mean) \
      / (class::size() - 1.0f); \
  printf("Mean = %f,   Variance = %f,   stddev = %f\n", \
         mean, variance, sqrt(variance)); \
}

__global__ void print_histograms_1() {
  PRINT_HISTORAM(Car, velocity, 50);
  PRINT_HISTORAM(Car, max_velocity, 50);
}

__global__ void print_histograms_2() {
  PRINT_HISTORAM(SharedSignalGroup, num_cells, 50);
  PRINT_HISTORAM(TrafficLight, num_signal_groups, 50);
}

__global__ void print_histograms_3() {
  PRINT_HISTORAM(Cell, num_incoming_cells, 50);
}

__global__ void print_histograms_4() {
  PRINT_HISTORAM(Cell, num_outgoing_cells, 50);
}

__global__ void print_histograms_5() {
  PRINT_HISTORAM(Cell, street_max_velocity, 50);
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
