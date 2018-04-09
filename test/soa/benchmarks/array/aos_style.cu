#define NDEBUG

#define NUM_INST 524288
#define ARRAY_SIZE 48

#include "executor/cuda_executor.h"
#include "soa/soa.h"

#include "benchmark.h"

using ikra::soa::IndexType;
using ikra::soa::SoaLayout;
using ikra::executor::cuda::construct;

#define CUDA_THREAD_ID (threadIdx.x + blockIdx.x * blockDim.x)

class DummyClass : public SoaLayout<DummyClass, NUM_INST> {
 public:
  IKRA_INITIALIZE_CLASS

  __device__ DummyClass(int f0, int f2): field0(f0), field2(f2) {
    for (int i = 0; i < ARRAY_SIZE; ++i) {
      field1[i] = CUDA_THREAD_ID*17 + i;
    }
  }

  int_ field0;

  // Array has size 12 bytes.
  array_(int, ARRAY_SIZE, soa) field1;

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
  printf("Time for action: %lu\n", time_action);

#ifndef NDEBUG
  // Check result (some samples).
  for (int k = 0; k < 100; ++k) {
    int i = rand() % NUM_INST;
    for (int j = 0; j < ARRAY_SIZE; ++j) {
      int actual1 = DummyClass::get(i)->field1[j];
      int expected1 = i*17 + j + 19 + 29 + 1357;
      if (actual1 != expected1) {
        printf("Wrong result! Expected %i, but found %i\n",
               expected1, actual1);
        exit(1);
      }
    }
  }
#endif  // NDEBUG
}

int main() {
  run_test_construct_and_execute();
}
