#include <array>

#include "gtest/gtest.h"

#include "executor/cuda_executor.h"
#include "soa/soa.h"

using ikra::soa::IndexType;
using ikra::soa::SoaLayout;
using ikra::executor::cuda::construct;
using ikra::executor::cuda::KernelConfig;
using ikra::executor::cuda::KernelConfiguration;

const static int kTestSize = 12;
const static int kInnerArraySize = 4;

class DummyClass : public SoaLayout<DummyClass, 1000> {
 public:
  IKRA_INITIALIZE_CLASS

  __host__ __device__ DummyClass(int f0, int f1) : field0(f0), field1(f1) {
    // Initialize array.
    for (int i = 0; i < kInnerArraySize; ++i) {
      nested_array_[i] = 100*id() + i;
    }
  }

  int_ field0;
  int_ field1;

  array_(int, kInnerArraySize, fully_inlined) nested_array_;

  template<int Ikra_VW_SZ>
  __device__ void add_fields(int increment) {
    field0 = field0 + field1 + increment + this->id() + Ikra_VW_SZ;
  }

  template<int Ikra_VW_SZ>
  __device__ void add_fields_no_arg() {
    field0 = field0 + field1 + this->id() + Ikra_VW_SZ;
  }

  template<int Ikra_VW_SZ>
  __device__ void update_array_vw(int val) {
    for (int& el : nested_array_.vw_iterator<Ikra_VW_SZ>()) {
      el += 10 + val;
    }
  }
};

IKRA_DEVICE_STORAGE(DummyClass);


// Cannot run "cuda_execute" inside gtest case.
void run_test_outer_cuda_execute_configuration() {
  DummyClass::initialize_storage();
  EXPECT_EQ(DummyClass::size(), 0UL);

  DummyClass* first = construct<DummyClass>(kTestSize, 5, 6);
  gpuErrchk(cudaPeekAtLastError());

  // Use a virtual warp size of 4.
  cuda_execute_vw(&DummyClass::add_fields, KernelConfiguration<4>(12),
                  first, 12, 10);

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    int actual = DummyClass::get(i)->field0;
    int expected = 10 + 5 + 6 + i + 4;
    EXPECT_EQ(actual, expected);
  }

  // Copy size to host memory and compare.
  EXPECT_EQ(DummyClass::size(), static_cast<IndexType>(kTestSize));

  // Make sure that we had no CUDA failures.
  gpuErrchk(cudaPeekAtLastError());
}

void run_test_outer_cuda_execute_configuration_no_arg() {
  DummyClass::initialize_storage();
  EXPECT_EQ(DummyClass::size(), 0UL);

  DummyClass* first = construct<DummyClass>(kTestSize, 5, 6);
  gpuErrchk(cudaPeekAtLastError());

  // Use a virtual warp size of 4.
  cuda_execute_vw(&DummyClass::add_fields_no_arg, KernelConfiguration<4>(12),
                  first, 12);

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    int actual = DummyClass::get(i)->field0;
    int expected = 5 + 6 + i + 4;
    EXPECT_EQ(actual, expected);
  }

  // Copy size to host memory and compare.
  EXPECT_EQ(DummyClass::size(), static_cast<IndexType>(kTestSize));

  // Make sure that we had no CUDA failures.
  gpuErrchk(cudaPeekAtLastError());
}

void run_test_range_based_for_loop() {
  DummyClass::initialize_storage();
  EXPECT_EQ(DummyClass::size(), 0UL);

  DummyClass* first = construct<DummyClass>(kTestSize, 5, 6);
  gpuErrchk(cudaPeekAtLastError());

  // Use a virtual warp size of 4.
  cuda_execute_vw(&DummyClass::update_array_vw, KernelConfiguration<4>(12),
                  first, 12, 20);

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    for (int j = 0; j < kInnerArraySize; ++j) {
      int actual = DummyClass::get(i)->nested_array_[j];
      int expected = 100*i + j + 10 + 20;
      EXPECT_EQ(actual, expected);
    }
  }

  // Copy size to host memory and compare.
  EXPECT_EQ(DummyClass::size(), static_cast<IndexType>(kTestSize));

  // Make sure that we had no CUDA failures.
  gpuErrchk(cudaPeekAtLastError());
}

void run_test_outer_cuda_execute_strategy() {
  DummyClass::initialize_storage();
  EXPECT_EQ(DummyClass::size(), 0UL);

  DummyClass* first = construct<DummyClass>(kTestSize, 5, 6);
  gpuErrchk(cudaPeekAtLastError());

  cuda_execute_vw(&DummyClass::add_fields, KernelConfig<>::standard(),
                  first, 12, 10);

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    int actual = DummyClass::get(i)->field0;
    int expected = 10 + 5 + 6 + i + 1;
    EXPECT_EQ(actual, expected);
  }

  // Copy size to host memory and compare.
  EXPECT_EQ(DummyClass::size(), static_cast<IndexType>(kTestSize));

  // Make sure that we had no CUDA failures.
  gpuErrchk(cudaPeekAtLastError());
}

void run_test_outer_cuda_execute_none() {
  DummyClass::initialize_storage();
  EXPECT_EQ(DummyClass::size(), 0UL);

  DummyClass* first = construct<DummyClass>(kTestSize, 5, 6);
  gpuErrchk(cudaPeekAtLastError());

  cuda_execute_vw(&DummyClass::add_fields,
                  first, 12, 10);

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    int actual = DummyClass::get(i)->field0;
    int expected = 10 + 5 + 6 + i + 1;
    EXPECT_EQ(actual, expected);
  }

  // Copy size to host memory and compare.
  EXPECT_EQ(DummyClass::size(), static_cast<IndexType>(kTestSize));

  // Make sure that we had no CUDA failures.
  gpuErrchk(cudaPeekAtLastError());
}

TEST(VirtualWarpTest, TestOuterCudaExecuteStrategy) {
  run_test_outer_cuda_execute_strategy();
}

TEST(VirtualWarpTest, TestOuterCudaExecuteStrategyNoArg) {
  // Call member function with no argument.
  run_test_outer_cuda_execute_configuration_no_arg();
}

TEST(VirtualWarpTest, TestOuterCudaExecuteNone) {
  run_test_outer_cuda_execute_none();
}

TEST(VirtualWarpTest, TestOuterCudaExecuteConfiguration) {
  run_test_outer_cuda_execute_configuration();
}

TEST(VirtualWarpTest, TestRangeBasedForLoop) {
  run_test_range_based_for_loop();
}

