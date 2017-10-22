#include "gtest/gtest.h"
#include "../../soa/soa.h"

namespace {
using ikra::soa::SoaLayout;
using ikra::soa::kAddressModeZero;

static const int kClassMaxInst = 1024;
static const int kTestSize = 9;

// Zero addressing mode.
#define IKRA_TEST_CLASSNAME TestClassZ
#define IKRA_TEST_ADDRESS_MODE kAddressModeZero
#include "array_test_layout.inc"
#undef IKRA_TEST_CLASSNAME
#undef IKRA_TEST_ADDRESS_MODE

// Valid addressing mode.
#define IKRA_TEST_CLASSNAME TestClassV
#define IKRA_TEST_ADDRESS_MODE sizeof(int)
#include "array_test_layout.inc"
#undef IKRA_TEST_CLASSNAME
#undef IKRA_TEST_ADDRESS_MODE

template<typename T>
class ArrayTest : public testing::Test {};

TYPED_TEST_CASE_P(ArrayTest);

TYPED_TEST_P(ArrayTest, AosArray) {
  TypeParam** instances = new TypeParam*[kTestSize];

  // Create a few instances
  for (int i = 0; i < kTestSize; ++i) {
    instances[i] = new TypeParam();
    instances[i]->field0 = i + 1;
    EXPECT_EQ(instances[i]->field0, i + 1);
    instances[i]->field2 = i*i;
  }

  // Set values.
  for (int i = 0; i < kTestSize; ++i) {
    instances[i]->field1[0] = 1*i + 1;
    instances[i]->field1[1] = 2*i + 2;
    // TODO: Should use "." for consistency.
    instances[i]->field1->at(2) = 3*i + 3;
  }

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    // Ensure that no data was overwritten.
    EXPECT_EQ(instances[i]->field0, i + 1);
    EXPECT_EQ(instances[i]->field2, i*i);

    int expected = i+1 + 2*i+2 + 3*i+3;
    EXPECT_EQ(instances[i]->field1_sum(), expected);
  }
}

TYPED_TEST_P(ArrayTest, SoaArray) {
  TypeParam** instances = new TypeParam*[kTestSize];

  // Create a few instances
  for (int i = 0; i < kTestSize; ++i) {
    instances[i] = new TypeParam();
  }

  // Set values.
  for (int i = 0; i < kTestSize; ++i) {
    instances[i]->field3[0] = 1*i + 1;
    instances[i]->field3[1] = 2*i + 2;
    instances[i]->field3->at(2) = 3*i + 3;
  }

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    int expected = i+1 + 2*i+2 + 3*i+3;
    EXPECT_EQ(instances[i]->field3_sum(), expected);
  }
}

TYPED_TEST_P(ArrayTest, SoaArrayStaticGet) {
  TypeParam** instances = new TypeParam*[kTestSize];

  // Create a few instances
  for (int i = 0; i < kTestSize; ++i) {
    instances[i] = new TypeParam();
  }

  // Set values.
  for (int i = 0; i < kTestSize; ++i) {
    instances[i]->field3.template at<0>() = 1*i + 1;
    instances[i]->field3.template at<1>() = 2*i + 2;
    instances[i]->field3.template at<2>() = 3*i + 3;
  }

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    int expected = i+1 + 2*i+2 + 3*i+3;
    EXPECT_EQ(instances[i]->field3_sum(), expected);
  }
}

REGISTER_TYPED_TEST_CASE_P(ArrayTest,
                           AosArray,
                           SoaArray,
                           SoaArrayStaticGet);

INSTANTIATE_TYPED_TEST_CASE_P(Valid, ArrayTest, TestClassV);
INSTANTIATE_TYPED_TEST_CASE_P(Zero, ArrayTest, TestClassZ);

}  // namespace
