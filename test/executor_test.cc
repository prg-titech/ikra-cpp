#include "gtest/gtest.h"
#include "executor/executor.h"
#include "executor/iterator.h"
#include "soa/soa.h"

namespace {
using ikra::soa::SoaLayout;
using ikra::soa::kAddressModeZero;
using ikra::executor::execute;
using ikra::executor::make_iterator;

static const int kClassMaxInst = 1024;
static const int kTestSize = 40;

char storage_buffer[100000];

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
  TypeParam::initialize_storage();
  std::array<TypeParam*, kTestSize> arr;

  for (int i = 0; i < kTestSize; ++i) {
    arr[i] = new TypeParam(i + 1, 100*i + 1);
  }

  execute(arr.begin(), arr.end(), &TypeParam::add_field1_and_a_to_field0, 50);

  // Check result
  for (int i = 0; i < kTestSize; ++i) {
    int expected = i + 1 + 100*i + 1 + 50;
    EXPECT_EQ(arr[i]->field0, expected);
  }
}

TYPED_TEST_P(ExecutorTest, IteratorFromPointers) {
  TypeParam::initialize_storage();
  std::array<TypeParam*, kTestSize> arr;

  for (int i = 0; i < kTestSize; ++i) {
    arr[i] = new TypeParam(i + 1, 200*i + 1);
  }

  execute(make_iterator(arr[0]),
          ++make_iterator(arr[kTestSize - 1]),
          &TypeParam::add_field1_and_a_to_field0,
          50);

  // Check result
  for (int i = 0; i < kTestSize; ++i) {
    int expected = i + 1 + 200*i + 1 + 50;
    EXPECT_EQ(arr[i]->field0, expected);
  }
}

TYPED_TEST_P(ExecutorTest, ExecuteAndReduce) {
  TypeParam::initialize_storage();
  std::array<TypeParam*, kTestSize> arr;

  for (int i = 0; i < kTestSize; ++i) {
    arr[i] = new TypeParam(i, 100*i);
  }

  int expected = ((kTestSize - 1)*kTestSize)/2*101 + kTestSize;
  int actual = execute_and_reduce(make_iterator(arr[0]),
                                  ++make_iterator(arr[kTestSize - 1]),
                                  &TypeParam::sum_plus_delta,
                                  [](int a, int b) { return a + b; }, 0, 1);

  EXPECT_EQ(actual, expected);
}

REGISTER_TYPED_TEST_CASE_P(ExecutorTest,
                           StdArray,
                           IteratorFromPointers,
                           ExecuteAndReduce);

INSTANTIATE_TYPED_TEST_CASE_P(Valid, ExecutorTest, TestClassV);
INSTANTIATE_TYPED_TEST_CASE_P(Zero, ExecutorTest, TestClassZ);

}  // namespace

