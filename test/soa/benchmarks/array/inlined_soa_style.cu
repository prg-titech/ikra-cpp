#include "executor/cuda_executor.h"
#include "soa/soa.h"

#include "benchmark.h"
#include "configuration.h"

using ikra::soa::IndexType;
using ikra::soa::SoaLayout;
using ikra::executor::cuda::construct;
using ikra::soa::StaticStorageWithArena;
using ikra::soa::kAddressModeZero;

#define READ_FROM_ARENA_ELEMENTS 128
#define EXTRA_BYTES (READ_FROM_ARENA_ELEMENTS*sizeof(int)*NUM_INST)
#define INLINE_ARR_SIZE (ARRAY_SIZE - READ_FROM_ARENA_ELEMENTS)

class DummyClass : public SoaLayout<DummyClass, NUM_INST, kAddressModeZero,
    StaticStorageWithArena<EXTRA_BYTES>> {
 public:
  IKRA_INITIALIZE_CLASS

  __device__ DummyClass(int f0, int f2): field0(f0), field2(f2),
                                         field1(ARRAY_SIZE) {
    for (int i = 0; i < ARRAY_SIZE; ++i) {
      field1[i] = id()*17 + i;
    }
  }

  int_ field0;

  array_(int, INLINE_ARR_SIZE, inline_soa) field1;

  int_ field2;

  __device__ void update_field1(int increment) {
    for (int i = 0; i < ARRAY_SIZE; ++i) {
      field1[i] += increment + field0 + field2;
    }
  }
};

IKRA_DEVICE_STORAGE(DummyClass);


void action() {
  DummyClass::initialize_storage();
  DummyClass* first = construct<DummyClass>(NUM_INST, 29, 1357);
  gpuErrchk(cudaPeekAtLastError());

  cuda_execute(&DummyClass::update_field1, first, NUM_INST, 19);
  gpuErrchk(cudaPeekAtLastError());

  cudaDeviceSynchronize();
}

void run_test_construct_and_execute() {
  uint64_t time_action = measure<>::execution(action);
  printf("[INLINE-%i/%i SOA] Time for action: %lu\n",
         INLINE_ARR_SIZE, ARRAY_SIZE, time_action);

#ifndef NDEBUG
  // Check result (some samples).
  for (int k = 0; k < 100; ++k) {
    int i = rand() % NUM_INST;
    for (int j = 0; j < ARRAY_SIZE; ++j) {
      int actual1 = DummyClass::get(i)->field1[j];
      int expected1 = i*17 + j + 19 + 29 + 1357;
      if (actual1 != expected1) {
        printf("[INLINE] Dummy[%i].field1[%i]: Expected %i, but found %i\n",
               i, j, expected1, actual1);
        exit(1);
      }
    }
  }
#endif  // NDEBUG
}

int main() {
  run_test_construct_and_execute();
}
