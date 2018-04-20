#ifndef EXAMPLE_TRAFFIC_CONFIGURATION_H
#define EXAMPLE_TRAFFIC_CONFIGURATION_H

using ikra::soa::kLayoutModeSoa;
using ikra::soa::kLayoutModeAos;

class Cell;
class SharedSignalGroup;

// Simulation logic.
#define SMART_TRAFFIC_LIGHT
#define MOVE_ONTO_LARGE_STREETS

// Statistics for benchmark.
static const uint32_t kNumIterations = 1000;
static const uint32_t kNumBenchmarkRuns = 5;

// Number of objects.
#define LARGE_DATASET
#ifdef LARGE_DATASET
// Adapted for dataset: OSM-urbanized, Denver_Aurora_CO
static const uint32_t kNumCells = 8339802;
const uint32_t kNumCars = 229376;
static const uint32_t kNumSharedSignalGroups = 205129;
static const uint32_t kNumTrafficLights = 62819;
static const uint32_t kNumPriorityCtrls = 4792;
#else
// Small dataset can run on my laptop.
static const uint32_t kNumCells = 950778;
const uint32_t kNumCars = 50000;
static const uint32_t kNumSharedSignalGroups = 344418;
static const uint32_t kNumTrafficLights = 8000;
static const uint32_t kNumPriorityCtrls = 1000;
#endif  // LARGE

// Maximum array sizes.
static const uint32_t kMaxNumCellIncoming = 8;
static const uint32_t kMaxNumCellOutgoing = 8;
static const uint32_t kMaxNumCarPath = 30;
static const uint32_t kMaxNumSharedSignalGroupCells = 8;
static const uint32_t kMaxNumTrafficLightGroups = 6;

// Manual padding for AOS mode.
static const uint32_t kCellAlignmentPadding = 2;
static const uint32_t kCarAlignmentPadding = 7;
static const uint32_t kSharedSignalGroupAlignmentPadding = 4;
static const uint32_t kPriorityCtrlAlignmentPadding = 4;
// No padding required for traffic lights.
// static const uint32_t kTrafficLightAlignmentPadding = 0;


#if BENCHMARK_MODE != 1
// Dummy configuration
#define BENCH_CELL_IN_MODE -1
#define BENCH_CELL_OUT_MODE -1
#define BENCH_CAR_MODE -1
#define BENCH_SIGNAL_GROUP_MODE -1
#define BENCH_TRAFFIC_LIGHT_MODE -1
#define BENCH_LAYOUT_MODE kLayoutModeSoa
#endif

// Layout mode: SOA or AOS.
static const int kCellLayoutMode = BENCH_LAYOUT_MODE;
static const int kCarLayoutMode = BENCH_LAYOUT_MODE;
static const int kSharedSignalGroupLayoutMode = BENCH_LAYOUT_MODE;
static const int kPriorityCtrlLayoutMode = BENCH_LAYOUT_MODE;
static const int kTrafficLightLayoutMode = BENCH_LAYOUT_MODE;

// Array configurations.
#if BENCH_CELL_IN_MODE == -1
#define ARRAY_CELL_INCOMING \
    array_(Cell*, kMaxNumCellIncoming, object)
#elif BENCH_CELL_IN_MODE == -2
#define ARRAY_CELL_INCOMING \
    array_(Cell*, kMaxNumCellIncoming, fully_inlined)
#elif !defined(BENCH_CELL_IN_MODE)
    #error BENCH_CELL_IN_MODE is undefined
#else
static const uint32_t kArrInlineIncomingCells = BENCH_CELL_IN_MODE;   // 4
#define ARRAY_CELL_INCOMING \
    array_(Cell*, kArrInlineIncomingCells, partially_inlined)
#define ARRAY_CELL_IN_IS_PARTIAL
#endif

#if BENCH_CELL_OUT_MODE == -1
#define ARRAY_CELL_OUTGOING \
    array_(Cell*, kMaxNumCellOutgoing, object)
#elif BENCH_CELL_OUT_MODE == -2
#define ARRAY_CELL_OUTGOING \
    array_(Cell*, kMaxNumCellOutgoing, fully_inlined)
#elif !defined(BENCH_CELL_OUT_MODE)
    #error BENCH_CELL_OUT_MODE is undefined
#else
static const uint32_t kArrInlineOutgoingCells = BENCH_CELL_OUT_MODE;   // 4
#define ARRAY_CELL_OUTGOING \
    array_(Cell*, kArrInlineOutgoingCells, partially_inlined)
#define ARRAY_CELL_OUT_IS_PARTIAL
#endif

#if BENCH_CAR_MODE == -1
#define ARRAY_CAR_PATH \
    array_(Cell*, kMaxNumCarPath, object)
#elif BENCH_CAR_MODE == -2
#define ARRAY_CAR_PATH \
    array_(Cell*, kMaxNumCarPath, fully_inlined)
#elif !defined(BENCH_CAR_MODE)
    #error BENCH_CAR_MODE is undefined
#else
static const uint32_t kArrInlinePath = BENCH_CAR_MODE;   // 5
#define ARRAY_CAR_PATH \
    array_(Cell*, kArrInlinePath, partially_inlined)
#define ARRAY_CAR_IS_PARTIAL
#endif

#if BENCH_SIGNAL_GROUP_MODE == -1
#define ARRAY_SIGNAL_GROUP_CELLS \
    array_(Cell*, kMaxNumSharedSignalGroupCells, object)
#elif BENCH_SIGNAL_GROUP_MODE == -2
#define ARRAY_SIGNAL_GROUP_CELLS \
    array_(Cell*, kMaxNumSharedSignalGroupCells, fully_inlined)
#elif !defined(BENCH_SIGNAL_GROUP_MODE)
    #error BENCH_SIGNAL_GROUP_MODE is undefined
#else
static const uint32_t kArrInlineSignalGroupCells =
    BENCH_SIGNAL_GROUP_MODE;    // 4
#define ARRAY_SIGNAL_GROUP_CELLS \
    array_(Cell*, kArrInlineSignalGroupCells, partially_inlined)
#define ARRAY_SIGNAL_GROUP_IS_PARTIAL
#endif

#if BENCH_TRAFFIC_LIGHT_MODE == -1
#define ARRAY_TRAFFIC_LIGHT_SIGNAL_GROUPS \
    array_(SharedSignalGroup*, kMaxNumTrafficLightGroups, object)
#elif BENCH_TRAFFIC_LIGHT_MODE == -2
#define ARRAY_TRAFFIC_LIGHT_SIGNAL_GROUPS \
    array_(SharedSignalGroup*, kMaxNumTrafficLightGroups, fully_inlined)
#elif !defined(BENCH_TRAFFIC_LIGHT_MODE)
    #error BENCH_TRAFFIC_LIGHT_MODE is undefined
#else
static const uint32_t kArrInlineTrafficLightSignalGroups =
    BENCH_TRAFFIC_LIGHT_MODE;    // 4
#define ARRAY_TRAFFIC_LIGHT_SIGNAL_GROUPS \
    array_(SharedSignalGroup*, kArrInlineTrafficLightSignalGroups, \
           partially_inlined)
#define ARRAY_TRAFFIC_LIGHT_IS_PARTIAL
#endif

// Sizes of arena.
// Worse case: avg = 1.05 each, so share 3 slots/instance.
static const uint32_t kCellArenaSize = kNumCells*3*sizeof(Cell*);
// Worse case: avg is around 20.
static const uint32_t kCarArenaSize = kNumCars*21*sizeof(Cell*);
static const uint32_t kSharedSignalGroupArenaSize =
    kNumSharedSignalGroups*4*sizeof(Cell*);
// Worse case: ~40% have 4 signal groups or more.
static const uint32_t kTrafficLightArenaSize =
    kNumTrafficLights*3*sizeof(SharedSignalGroup*);
static const uint32_t kPriorityCtrlArenaSize = 0;

// Types of other arrays.
// Priority controller have always two signal groups.
#define ARRAY_PRIORITY_CTRL_SIGNAL_GROUPS \
    array_(SharedSignalGroup*, 2, fully_inlined)

#endif  // EXAMPLE_TRAFFIC_CONFIGURATION_H
