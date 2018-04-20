#include <array>

#include "gtest/gtest.h"

#include "executor/cuda_executor.h"
#include "soa/soa.h"

using ikra::soa::IndexType;
using ikra::soa::SoaLayout;
using ikra::executor::cuda::construct;

const static int kTestSize = 12;

#define CUDA_THREAD_ID (threadIdx.x + blockIdx.x * blockDim.x)

class DummyClass : public SoaLayout<DummyClass, 1000> {
 public:
  IKRA_INITIALIZE_CLASS

  __device__ DummyClass(int v2, int v4) : field2(v2), field4(v4) {
    for (int i = 0; i < 3; ++i) {
      field1[i] = CUDA_THREAD_ID*1000 + i*31;
      field3[i] = CUDA_THREAD_ID*500 + i*59;
    }
  }

  int_ field0;

  // Array has size 12 bytes.
  array_(int, 3, object) field1;
  int_ field2;

  // Array has size 12 bytes.
  array_(int, 3, fully_inlined) field3;
  int_ field4;

  __device__ void update_field1(int increment) {
    for (int i = 0; i < 3; ++i) {
      field1[i] += field2 + increment;
    }
  }

  __device__ void update_field3(int increment) {
    for (int i = 0; i < 3; ++i) {
      field3[i] += field4 + increment;
    }
  }

  __device__ void update_field3_second() {
    for (int i = 0; i < 3; ++i) {
      field3[i] += field1[i];
    }
  }
};

IKRA_DEVICE_STORAGE(DummyClass);


// Cannot run "cuda_execute" inside gtest case.
void run_test_construct_and_execute() {
  DummyClass::initialize_storage();
  EXPECT_EQ(DummyClass::size(), 0UL);

  DummyClass* first = construct<DummyClass>(kTestSize, 17, 29);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaPeekAtLastError());

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    for (int j = 0; j < 3; ++j) {
      int actual1 = DummyClass::get(i)->field1[j];
      int actual3 = DummyClass::get(i)->field3[j];
      int expected1 = i*1000 + j*31;
      int expected3 = i*500 + j*59;
      EXPECT_EQ(actual1, expected1);
      EXPECT_EQ(actual3, expected3);
    }
  }

  cuda_execute(&DummyClass::update_field1, first, kTestSize, 2);
  gpuErrchk(cudaPeekAtLastError());

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    for (int j = 0; j < 3; ++j) {
      int actual1 = DummyClass::get(i)->field1[j];
      int actual3 = DummyClass::get(i)->field3[j];
      int expected1 = i*1000 + j*31 + 2 + 17;
      int expected3 = i*500 + j*59;
      EXPECT_EQ(actual1, expected1);
      EXPECT_EQ(actual3, expected3);
    }
  }

  cuda_execute(&DummyClass::update_field3, first, kTestSize, 5);
  gpuErrchk(cudaPeekAtLastError());

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    for (int j = 0; j < 3; ++j) {
      int actual1 = DummyClass::get(i)->field1[j];
      int actual3 = DummyClass::get(i)->field3[j];
      int expected1 = i*1000 + j*31 + 2 + 17;
      int expected3 = i*500 + j*59 + 5 + 29;
      EXPECT_EQ(actual1, expected1);
      EXPECT_EQ(actual3, expected3);
    }
  }

  cuda_execute(&DummyClass::update_field3_second, first, kTestSize);
  gpuErrchk(cudaPeekAtLastError());

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    for (int j = 0; j < 3; ++j) {
      int actual1 = DummyClass::get(i)->field1[j];
      int actual3 = DummyClass::get(i)->field3[j];
      int expected1 = i*1000 + j*31 + 2 + 17;
      int expected3 = i*500 + j*59 + 5 + 29 + expected1;
      EXPECT_EQ(actual1, expected1);
      EXPECT_EQ(actual3, expected3);
    }
  }

  // Copy size to host memory and compare.
  EXPECT_EQ(DummyClass::size(), static_cast<IndexType>(kTestSize));

  // Make sure that we had no CUDA failures.
  gpuErrchk(cudaPeekAtLastError());
}


TEST(CudaArrayTest, ConstructAndExecute) {
  run_test_construct_and_execute();
}

