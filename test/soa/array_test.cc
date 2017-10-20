#include "gtest/gtest.h"
#include "../../soa/soa.h"

namespace {
using ikra::soa::SoaLayout;
using ikra::soa::kAddressModeZero;

static const int kClassMaxInst = 1024;
static const int kTestSize = 9;

// Note: All the extra keywords (typename etc.) are required because this
// class is templatized.
template<int AddressMode>
class TestClass : public SoaLayout<TestClass<AddressMode>, 36,
                                   kClassMaxInst, AddressMode> {
 public:
  using SelfSuper = SoaLayout<TestClass<AddressMode>, 36,
                              kClassMaxInst, AddressMode>;

  static typename SelfSuper::Storage storage;

  typename SelfSuper::template int_<0> field0;

  // Array has size 12 bytes.
  typename SelfSuper::template aos_array_<int, 3, 4> field1;
  typename SelfSuper::template int_<16> field2;

  // Array has size 12 bytes.
  typename SelfSuper::template soa_array_<int, 3, 20> field3;

  typename SelfSuper::template int_<32> field4;

  int field1_sum() {
    int result = 0;
    for (int i = 0; i < 3; ++i) {
      result = result + field1[i];
    }
    return result;
  }

  int field3_sum() {
    int result = 0;
    for (int i = 0; i < 3; ++i) {
      result = result + field3[i];
    }
    return result;
  }
};

template<>
TestClass<sizeof(int)>::Storage TestClass<sizeof(int)>::storage;

template<>
TestClass<kAddressModeZero>::Storage TestClass<kAddressModeZero>::storage;

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

INSTANTIATE_TYPED_TEST_CASE_P(Valid, ArrayTest,
                              TestClass<sizeof(int)>);
INSTANTIATE_TYPED_TEST_CASE_P(Zero, ArrayTest,
                              TestClass<kAddressModeZero>);

}  // namespace
