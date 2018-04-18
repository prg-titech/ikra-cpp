#ifndef EXAMPLE_TRAFFIC_CONFIGURATION_H
#define EXAMPLE_TRAFFIC_CONFIGURATION_H

class Cell;
class SharedSignalGroup;

// Statistics for benchmark
static const uint32_t kNumIterations = 2000;
static const uint32_t kNumBenchmarkRuns = 5;

// Number of objects.
static const uint32_t kNumCells = 5856393;
static const uint32_t kNumCars = 250000;
static const uint32_t kNumSharedSignalGroups = 344418;
static const uint32_t kNumTrafficLights = 50972;
static const uint32_t kNumPriorityCtrls = 6000;

// Inlining sizes of arrays.
static const uint32_t kArrInlineOutgoingCells = 4;
static const uint32_t kArrInlineIncomingCells = 8;
static const uint32_t kArrInlinePath = 10;
static const uint32_t kArrInlineSignalGroupCells = 4;
static const uint32_t kArrInlineTrafficLightSignalGroups = 4;
static const uint32_t kArrInlinePriorityCtrlSignalGroups = 4;

// Sizes of arena.
static const uint32_t kCellArenaSize = kNumCells*4*sizeof(Cell*);
static const uint32_t kCarArenaSize = kNumCars*10*sizeof(Cell*);
static const uint32_t kSharedSignalGroupArenaSize =
    kNumSharedSignalGroups*4*sizeof(Cell*);
static const uint32_t kTrafficLightArenaSize =
    kNumTrafficLights*8*sizeof(SharedSignalGroup*);
static const uint32_t kPriorityCtrlArenaSize =
    kNumPriorityCtrls*8*sizeof(SharedSignalGroup*);

// Kind of arrays.
#define ARRAY_CELL_IS_PARTIAL
#define ARRAY_CAR_IS_PARTIAL
#define ARRAY_SIGNAL_GROUP_IS_PARTIAL
#define ARRAY_TRAFFIC_LIGHT_IS_PARTIAL
#define ARRAY_PRIORITY_CTRL_IS_PARTIAL

#define ARRAY_CELL_INCOMING \
    array_(Cell*, kArrInlineIncomingCells, inline_soa)
#define ARRAY_CELL_OUTGOING \
    array_(Cell*, kArrInlineOutgoingCells, inline_soa)
#define ARRAY_CAR_PATH \
    array_(Cell*, kArrInlinePath, inline_soa)
#define ARRAY_SIGNAL_GROUP_CELLS \
    array_(Cell*, kArrInlineSignalGroupCells, inline_soa)
#define ARRAY_TRAFFIC_LIGHT_SIGNAL_GROUPS \
    array_(SharedSignalGroup*, kArrInlineTrafficLightSignalGroups, inline_soa)
#define ARRAY_PRIORITY_CTRL_SIGNAL_GROUPS \
    array_(SharedSignalGroup*, kArrInlinePriorityCtrlSignalGroups, inline_soa)

#endif  // EXAMPLE_TRAFFIC_CONFIGURATION_H
