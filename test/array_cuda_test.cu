#include <array>

#include "gtest/gtest.h"

#include "executor/cuda_executor.h"
#include "soa/soa.h"

using ikra::soa::IndexType;
using ikra::soa::SoaLayout;
using ikra::executor::cuda::construct;

const static int kTestSize = 12;

class TestClass : public SoaLayout<TestClass, 1000> {
 public:
  IKRA_INITIALIZE_CLASS

  TestClass(std::array<double, 2> f0, std::array<double, 2> f1) {
    field0[0] = f0[0];
    field0[1] = f0[1];
    field1[0] = f1[0];
    field1[1] = f1[1];
  }

  array_(double, 2) field0;
  array_(double, 2) field1;

  __device__ void add_fields() {
    field0[0] = field0[0] + field0[1] + field1[0] + field1[1];
  }
};

IKRA_DEVICE_STORAGE(TestClass);


// Cannot run "cuda_execute" inside gtest case.
void run_test_set_from_host_and_read() {
  TestClass::initialize_storage();
  EXPECT_EQ(TestClass::size(), 0UL);

  for (int i = 0; i < kTestSize; ++i) {
    new TestClass({{ 1.0, 2.0 }}, {{ 3.0, 4.0 }});
  }
  gpuErrchk(cudaPeekAtLastError());

  TestClass* first = TestClass::get(0);
  cuda_execute(&TestClass::add_fields, first, kTestSize);

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    double actual = TestClass::get(i)->field0[0];
    double expected = 1.0 + 2.0 + 3.0 + 4.0;
    EXPECT_EQ(actual, expected);
  }

  // Copy size to host memory and compare.
  EXPECT_EQ(TestClass::size(), static_cast<IndexType>(kTestSize));

  // Make sure that we had no CUDA failures.
  gpuErrchk(cudaPeekAtLastError());
}

TEST(ArrayCudaTest, SetFromHostAndRead) {
  // This test ensures that address computation for array fields is done
  // correctly.
  run_test_set_from_host_and_read();
}
