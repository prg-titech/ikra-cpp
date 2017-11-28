#include <numeric>

#include "gtest/gtest.h"
#include "soa/soa.h"

namespace {
using namespace ikra::soa;

static const int kClassMaxInst = 1024;
static const int kTestSize = 9;

template<typename T, size_t N>
T array_sum(std::array<T, N> array) {
  return std::accumulate(array.begin(), array.end(), 0);
}

// Zero addressing mode.
#define IKRA_TEST_CLASSNAME TestClassZ
#define IKRA_TEST_ADDRESS_MODE kAddressModeZero
#include "array_test_layout.inc"
#undef IKRA_TEST_CLASSNAME
#undef IKRA_TEST_ADDRESS_MODE
IKRA_HOST_STORAGE(TestClassZ)

// Valid addressing mode.
#define IKRA_TEST_CLASSNAME TestClassV
#define IKRA_TEST_ADDRESS_MODE sizeof(int)
#include "array_test_layout.inc"
#undef IKRA_TEST_CLASSNAME
#undef IKRA_TEST_ADDRESS_MODE
IKRA_HOST_STORAGE(TestClassV)

template<typename T>
class ArrayTest : public testing::Test {};

TYPED_TEST_CASE_P(ArrayTest);

TYPED_TEST_P(ArrayTest, ArrayOperationsA) {
  TypeParam::initialize_storage();
  TypeParam** instances = new TypeParam*[kTestSize];
  std::array<std::array<int, 3>, kTestSize> arrays;

  // Initialize array fields and std::arrays.
  for (int i = 0; i < kTestSize; ++i) {
    instances[i] = new TypeParam();
    for (int j = 0; j < 3; ++j) {
      instances[i]->field1[j] = 2*i;
      instances[i]->field3[j] = i + j;
      arrays[i][j] = i - j;
    }
  }

  // Test operations between two std::arrays.
  for (int i = 0; i < kTestSize; ++i) {
    std::array<int, 3> addition = arrays[i] + arrays[i];
    EXPECT_EQ(array_sum(addition), 2 * array_sum(arrays[i]));
    std::array<int, 3> subtraction = addition - arrays[i];
    EXPECT_EQ(array_sum(subtraction), array_sum(arrays[i]));
    addition += arrays[i];
    EXPECT_EQ(array_sum(addition), 3 * array_sum(arrays[i]));
    subtraction -= arrays[i];
    EXPECT_EQ(array_sum(subtraction), 0);
  }

  // Test operations between std::arrays and AOS array fields.
  for (int i = 0; i < kTestSize; ++i) {
    std::array<int, 3> addition = arrays[i] + instances[i]->field1;
    EXPECT_EQ(array_sum(addition), 9*i - 3);
    std::array<int, 3> subtraction = instances[i]->field1 - arrays[i];
    EXPECT_EQ(array_sum(subtraction), 3*i + 3);
    addition += instances[i]->field1;
    EXPECT_EQ(array_sum(addition), 15*i - 3);
    subtraction -= instances[i]->field1;
    EXPECT_EQ(array_sum(subtraction), -3*i + 3);
  }

  // Test operations between std::arrays and SOA array fields.
  for (int i = 0; i < kTestSize; ++i) {
    std::array<int, 3> addition = arrays[i] + instances[i]->field3;
    EXPECT_EQ(array_sum(addition), 6*i);
    std::array<int, 3> subtraction = instances[i]->field3 - arrays[i];
    EXPECT_EQ(array_sum(subtraction), 6);
    addition += instances[i]->field3;
    EXPECT_EQ(array_sum(addition), 9*i + 3);
    subtraction -= instances[i]->field3;
    EXPECT_EQ(array_sum(subtraction), -3*i + 3);
  }

  // Test operations between AOS and SOA array fields.
  for (int i = 0; i < kTestSize; ++i) {
    std::array<int, 3> addition = instances[i]->field3 + instances[i]->field1;
    EXPECT_EQ(array_sum(addition), 9*i + 3);
    std::array<int, 3> subtraction = instances[i]->field1 - instances[i]->field3;
    EXPECT_EQ(array_sum(subtraction), 3*i - 3);
  }

  // Test operations between two AOS array fields.
  for (int i = 0; i < kTestSize; ++i) {
    std::array<int, 3> addition = instances[i]->field1 + instances[i]->field1;
    EXPECT_EQ(array_sum(addition), 12*i);
    std::array<int, 3> subtraction = instances[i]->field1 - instances[i]->field1;
    EXPECT_EQ(array_sum(subtraction), 0);
    instances[i]->field1 += instances[i]->field1;
    EXPECT_EQ(instances[i]->field1_sum(), 12*i);
    instances[i]->field1 -= instances[i]->field1;
    EXPECT_EQ(instances[i]->field1_sum(), 0);
  }

  // Test operations between two SOA array fields.
  for (int i = 0; i < kTestSize; ++i) {
    std::array<int, 3> addition = instances[i]->field3 + instances[i]->field3;
    EXPECT_EQ(array_sum(addition), 6*i + 6);
    std::array<int, 3> subtraction = instances[i]->field3 - instances[i]->field3;
    EXPECT_EQ(array_sum(subtraction), 0);
    instances[i]->field3 += instances[i]->field3;
    EXPECT_EQ(instances[i]->field3_sum(), 6*i + 6);
    instances[i]->field3 -= instances[i]->field3;
    EXPECT_EQ(instances[i]->field3_sum(), 0);
  }
}

TYPED_TEST_P(ArrayTest, ArrayOperationsM) {
  TypeParam::initialize_storage();
  TypeParam** instances = new TypeParam*[kTestSize];
  std::array<std::array<int, 3>, kTestSize> arrays;

  // Initialize array fields and std::arrays.
  for (int i = 0; i < kTestSize; ++i) {
    instances[i] = new TypeParam();
    for (int j = 0; j < 3; ++j) {
      instances[i]->field1[j] = 2*i;
      instances[i]->field3[j] = i + j;
      arrays[i][j] = i - j;
    }
  }

  // Test multiplications of std::arrays.
  for (int i = 0; i < kTestSize; ++i) {
    std::array<int, 3> mul1 = arrays[i] * 3;
    EXPECT_EQ(array_sum(mul1), 9*i - 9);
    std::array<int, 3> mul2 = 3 * arrays[i];
    EXPECT_EQ(array_sum(mul2), 9*i - 9);
    arrays[i] *= 2;
    EXPECT_EQ(array_sum(arrays[i]), 6*i - 6);
  }

  // Test multiplications of AOS array fields.
  for (int i = 0; i < kTestSize; ++i) {
    std::array<int, 3> mul1 = instances[i]->field1 * 3;
    EXPECT_EQ(array_sum(mul1), 18*i);
    std::array<int, 3> mul2 = 3 * instances[i]->field1;
    EXPECT_EQ(array_sum(mul2), 18*i);
    instances[i]->field1 *= 2;
    EXPECT_EQ(instances[i]->field1_sum(), 12*i);
  }

  // Test multiplications of AOS array fields.
  for (int i = 0; i < kTestSize; ++i) {
    std::array<int, 3> mul1 = instances[i]->field3 * 3;
    EXPECT_EQ(array_sum(mul1), 9*i + 9);
    std::array<int, 3> mul2 = 3 * instances[i]->field3;
    EXPECT_EQ(array_sum(mul2), 9*i + 9);
    instances[i]->field3 *= 2;
    EXPECT_EQ(instances[i]->field3_sum(), 6*i + 6);
  }
}

REGISTER_TYPED_TEST_CASE_P(ArrayTest, ArrayOperationsA, ArrayOperationsM);

INSTANTIATE_TYPED_TEST_CASE_P(Valid, ArrayTest, TestClassV);
INSTANTIATE_TYPED_TEST_CASE_P(Zero, ArrayTest, TestClassZ);

}  // namespace
