#include "gtest/gtest.h"
#include "executor/array.h"
#include "executor/iterator.h"
#include "soa/soa.h"

namespace {
using ikra::soa::SoaLayout;
using ikra::soa::kAddressModeZero;
using ikra::executor::IteratorExecutor;
using ikra::executor::Iterator;

static const int kClassMaxInst = 1024;
static const int kTestSize = 40;

// Zero addressing mode.
#define IKRA_TEST_CLASSNAME TestClassZ
#define IKRA_TEST_ADDRESS_MODE kAddressModeZero
#include "executor_test_layout.inc"
#undef IKRA_TEST_CLASSNAME
#undef IKRA_TEST_ADDRESS_MODE

// Valid addressing mode.
#define IKRA_TEST_CLASSNAME TestClassV
#define IKRA_TEST_ADDRESS_MODE sizeof(int)
#include "executor_test_layout.inc"
#undef IKRA_TEST_CLASSNAME
#undef IKRA_TEST_ADDRESS_MODE

template<typename T>
class ExecutorTest : public testing::Test {};

TYPED_TEST_CASE_P(ExecutorTest);

TYPED_TEST_P(ExecutorTest, StdArray) {
  std::array<TypeParam*, kTestSize> arr;

  for (int i = 0; i < kTestSize; ++i) {
    arr[i] = new TypeParam(i + 1, 100*i + 1);
  }

  auto executor = IteratorExecutor(arr.begin(), arr.end());
  executor.execute(&TypeParam::add_field1_and_a_to_field0, 50);

  // Check result
  for (int i = 0; i < kTestSize; ++i) {
    int expected = i + 1 + 100*i + 1 + 50;
    EXPECT_EQ(arr[i]->field0, expected);
  }
}

TYPED_TEST_P(ExecutorTest, IteratorFromPointers) {
  std::array<TypeParam*, kTestSize> arr;

  for (int i = 0; i < kTestSize; ++i) {
    arr[i] = new TypeParam(i + 1, 200*i + 1);
  }

  auto executor = IteratorExecutor(Iterator(arr[0]),
                                   ++Iterator(arr[kTestSize - 1]));
  executor.execute(&TypeParam::add_field1_and_a_to_field0, 50);

  // Check result
  for (int i = 0; i < kTestSize; ++i) {
    int expected = i + 1 + 200*i + 1 + 50;
    EXPECT_EQ(arr[i]->field0, expected);
  }
}

REGISTER_TYPED_TEST_CASE_P(ExecutorTest,
                           StdArray,
                           IteratorFromPointers);

INSTANTIATE_TYPED_TEST_CASE_P(Valid, ExecutorTest, TestClassV);
INSTANTIATE_TYPED_TEST_CASE_P(Zero, ExecutorTest, TestClassZ);

}  // namespace

