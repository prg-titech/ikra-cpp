#include "gtest/gtest.h"

#include "executor/cuda_executor.h"
#include "executor/executor.h"
#include "soa/soa.h"

using ikra::soa::IndexType;
using ikra::soa::SoaLayout;
using ikra::executor::cuda::construct;
using ikra::soa::StaticStorageWithArena;
using ikra::soa::kAddressModeZero;

// Size of inline SOA array.
#define ARRAY_SIZE 100
// Number of DummyClass instances.
#define NUM_INST 121
// Number of instances constructed on the device.
#define CONSTRUCT_DEVICE 79
// Number of instances constructed on the host.
#define CONSTRUCT_HOST (NUM_INST - CONSTRUCT_DEVICE)
// Number of array elements stored in the area.
#define READ_FROM_ARENA_ELEMENTS 32
// Size of the arena in bytes.
// TODO: Why is the +1 required here?
#define EXTRA_BYTES (READ_FROM_ARENA_ELEMENTS*sizeof(int)*NUM_INST+1)
// Number of array elements stored in inline storage.
#define INLINE_ARR_SIZE (ARRAY_SIZE - READ_FROM_ARENA_ELEMENTS)

class DummyClass : public SoaLayout<DummyClass, NUM_INST, kAddressModeZero,
    StaticStorageWithArena<EXTRA_BYTES>> {
 public:
  IKRA_INITIALIZE_CLASS

  __host__ __device__ DummyClass(int f0, int f2): field0(f0), field2(f2),
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


void run_test_construct_and_execute() {
  DummyClass::initialize_storage();
  // Create most instances on the device.
  DummyClass* first = construct<DummyClass>(CONSTRUCT_DEVICE, 29, 1357);
  gpuErrchk(cudaPeekAtLastError());

  // Create some instances on the host.
  ikra::executor::construct<DummyClass>(CONSTRUCT_HOST, 29, 1357);

  cuda_execute(&DummyClass::update_field1, first, NUM_INST, 19);
  gpuErrchk(cudaPeekAtLastError());

  // Check result.
  for (int i = 0; i < NUM_INST; ++i) {
    for (int j = 0; j < ARRAY_SIZE; ++j) {
      int actual1 = DummyClass::get(i)->field1[j];
      int expected1 = i*17 + j + 19 + 29 + 1357;
      EXPECT_EQ(actual1, expected1);
    }
  }
}

TEST(CudaInlineArrayMemcpyTest, ConstructAndExecute) {
  run_test_construct_and_execute();
}

