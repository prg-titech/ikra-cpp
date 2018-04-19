#ifndef EXAMPLE_TRAFFIC_CONFIGURATION_H
#define EXAMPLE_TRAFFIC_CONFIGURATION_H

class Cell;
class SharedSignalGroup;

// Simulation logic.
#define SMART_TRAFFIC_LIGHT
#define MOVE_ONTO_LARGE_STREETS

// Statistics for benchmark.
static const uint32_t kNumIterations = 1000;
static const uint32_t kNumBenchmarkRuns = 1;

// Number of objects.
static const uint32_t kNumCells = 11433334; //7112859; //6428104;
const uint32_t kNumCars = 250000;
static const uint32_t kNumSharedSignalGroups = 344418;
static const uint32_t kNumTrafficLights = 79437;
static const uint32_t kNumPriorityCtrls = 6000;

// Inlining sizes of arrays.
static const uint32_t kArrInlineOutgoingCells = 4;
static const uint32_t kArrInlineIncomingCells = 4;
static const uint32_t kArrInlinePath = 5;
static const uint32_t kArrInlineSignalGroupCells = 4;
static const uint32_t kArrInlineTrafficLightSignalGroups = 4;

// Sizes of arena.
static const uint32_t kCellArenaSize = kNumCells*4*sizeof(Cell*);
static const uint32_t kCarArenaSize = kNumCars*20*sizeof(Cell*);
static const uint32_t kSharedSignalGroupArenaSize =
    kNumSharedSignalGroups*4*sizeof(Cell*);
static const uint32_t kTrafficLightArenaSize =
    kNumTrafficLights*8*sizeof(SharedSignalGroup*);
static const uint32_t kPriorityCtrlArenaSize = 0;

// Types of arrays.
#define ARRAY_CELL_IS_PARTIAL
//#define ARRAY_CAR_IS_PARTIAL
#define ARRAY_SIGNAL_GROUP_IS_PARTIAL
#define ARRAY_TRAFFIC_LIGHT_IS_PARTIAL
// #define ARRAY_PRIORITY_CTRL_IS_PARTIAL

#define ARRAY_CELL_INCOMING \
    array_(Cell*, kArrInlineIncomingCells, partially_inlined)
#define ARRAY_CELL_OUTGOING \
    array_(Cell*, kArrInlineOutgoingCells, partially_inlined)
#define ARRAY_CAR_PATH \
    array_(Cell*, 16, object)
#define ARRAY_SIGNAL_GROUP_CELLS \
    array_(Cell*, kArrInlineSignalGroupCells, partially_inlined)
#define ARRAY_TRAFFIC_LIGHT_SIGNAL_GROUPS \
    array_(SharedSignalGroup*, kArrInlineTrafficLightSignalGroups, partially_inlined)

// Priority controller have always two signal groups.
#define ARRAY_PRIORITY_CTRL_SIGNAL_GROUPS \
    array_(SharedSignalGroup*, 2, fully_inlined)

#endif  // EXAMPLE_TRAFFIC_CONFIGURATION_H
