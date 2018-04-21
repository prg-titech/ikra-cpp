#include "executor/cuda_executor.h"
#include "executor/executor.h"
#include "soa/soa.h"

#include <chrono>
#include <limits>

template<typename TimeT = std::chrono::microseconds>
struct measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep execution(F&& func, Args&&... args)
    {
        auto start = std::chrono::steady_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast< TimeT> 
                            (std::chrono::steady_clock::now() - start);
        return duration.count();
    }
};

using ikra::soa::IndexType;
using ikra::soa::SoaLayout;
using ikra::soa::StaticStorageWithArena;
using ikra::soa::kAddressModeZero;

// Number of runs of this benchmark.
#define NUM_RUNS 10

#if BENCHMARK_MODE != 1
// Size of inline SOA array.
#define ARRAY_SIZE 64
// Number of DummyClass instances.
#define NUM_INST 32768
// Number of array elements stored in inline storage.
#define INLINE_SIZE 16
#endif

// Size of the arena in bytes.
// TODO: Why is the +1 required here?
#define EXTRA_BYTES ((ARRAY_SIZE - INLINE_SIZE)*sizeof(int)*NUM_INST+1)

// TODO: Alignment is broken in AOS mode.
class DummyClass : public SoaLayout<DummyClass, NUM_INST, kAddressModeZero,
    StaticStorageWithArena<EXTRA_BYTES>, ikra::soa::kLayoutModeAos> {
 public:
  IKRA_INITIALIZE_CLASS

  __host__ __device__ DummyClass(int f0,
                                 int array_size) : field0(f0),
                                                   field1(array_size),
                                                   array_size_(array_size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    assert(tid == id());

    for (int i = 0; i < array_size; ++i) {
      field1[i] = id()*17 + i;
    }
  }

  int_ field0;

  int_ array_size_;

  array_(int, INLINE_SIZE, partially_inlined) field1;

  __device__ void update_field1(int increment) {
    for (int i = 0; i < array_size_; ++i) {
      field1[i] += increment + field0;
    }
  }
};

IKRA_DEVICE_STORAGE(DummyClass);

// TODO: Cannot use construct<> here, because array size should be initialized
// with different values and this can only be done with field initializers.
__global__ void constructor_kernel(uint32_t num_objects) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_objects) {
    new(DummyClass::get_uninitialized(tid))
        DummyClass(29, ARRAY_SIZE/2 + tid%(ARRAY_SIZE/2));
  }

  if (tid == 0) {
    DummyClass::storage().increase_size(num_objects);
  }
}

void benchmark(DummyClass* first) {
  cuda_execute(&DummyClass::update_field1, first, NUM_INST, 19);
}

int main() {
  uint64_t best_time = std::numeric_limits<uint64_t>::max();

  for (int i = 0; i < NUM_RUNS; ++i) {
    DummyClass::initialize_storage();
    constructor_kernel<<<NUM_INST/1024+1, 1024>>>(NUM_INST);
    gpuErrchk(cudaPeekAtLastError());

    uint64_t new_time = measure<>::execution(
        benchmark, DummyClass::get_uninitialized(0));
    gpuErrchk(cudaPeekAtLastError());

    if (new_time < best_time) {
      best_time = new_time;
    }

    // Check some random values for correctness.
    for (int i = 0; i < 100; ++i) {
      auto obj_id = rand() % NUM_INST;
      auto* obj = DummyClass::get(obj_id);

      for (int j = 0; j < obj->array_size_; ++j) {
        int actual1 = obj->field1[j];
        int expected1 = obj_id*17 + j + 19 + 29;
        if (actual1 != expected1) {
          printf("Wrong result: obj[%i]->arr[%i]. Expected %i but found %i\n",
                 obj_id, j, expected1, actual1);
          exit(1);
        }
      }
    }
  }

  printf("%lu\n", (unsigned long) best_time);
}
