#include "gtest/gtest.h"
#include "soa/soa.h"

namespace {
using ikra::soa::SoaLayout;
using ikra::soa::StaticStorage;
using ikra::soa::kAddressModeZero;
using ikra::soa::kLayoutModeSoa;
using ikra::soa::kLayoutModeAos;

static const int kClassMaxInst = 1024;
static const int kTestSize = 40;

// Zero addressing mode.
#define IKRA_TEST_CLASSNAME TestClassZ_Soa
#define IKRA_TEST_ADDRESS_MODE kAddressModeZero
#define IKRA_TEST_LAYOUT_MODE kLayoutModeSoa
#include "basic_class_test_layout.inc"
#undef IKRA_TEST_CLASSNAME
#undef IKRA_TEST_ADDRESS_MODE
IKRA_HOST_STORAGE(TestClassZ_Soa)

#define IKRA_TEST_CLASSNAME TestClassZ_Aos
#define IKRA_TEST_ADDRESS_MODE kAddressModeZero
#define IKRA_TEST_LAYOUT_MODE kLayoutModeAos
#include "basic_class_test_layout.inc"
#undef IKRA_TEST_CLASSNAME
#undef IKRA_TEST_ADDRESS_MODE
IKRA_HOST_STORAGE(TestClassZ_Aos)

// Valid addressing mode.
#define IKRA_TEST_CLASSNAME TestClassV
#define IKRA_TEST_ADDRESS_MODE sizeof(int)
#define IKRA_TEST_LAYOUT_MODE kLayoutModeSoa
#include "basic_class_test_layout.inc"
#undef IKRA_TEST_CLASSNAME
#undef IKRA_TEST_ADDRESS_MODE
IKRA_HOST_STORAGE(TestClassV)


template<typename T>
class BasicClassTest : public testing::Test {};

TYPED_TEST_CASE_P(BasicClassTest);

TYPED_TEST_P(BasicClassTest, Fields) {
  TypeParam::initialize_storage();
  TypeParam** instances = new TypeParam*[kTestSize];

  // Create a few instances
  for (int i = 0; i < kTestSize; ++i) {
    instances[i] = new TypeParam();
  }

  // Set values.
  for (int i = 0; i < kTestSize; ++i) {
    instances[i]->field0 = i + 99;
    EXPECT_EQ(instances[i]->field0, i + 99);

    instances[i]->field4 = i*i;
    EXPECT_EQ(instances[i]->field4, i*i);

    instances[i]->field2 = i % 3 == 0;
    EXPECT_EQ(instances[i]->field2, i % 3 == 0);

    int expected = i % 3 == 0 ? i*i : -1;
    EXPECT_EQ(instances[i]->get_field4_if_field2(), expected);
  }

  // Do increment.
  for (int i = 0; i < kTestSize; ++i) {
    instances[i]->add_field0_to_field4();
  }

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    int expected = i % 3 == 0 ? i + i*i + 99: -1;
    EXPECT_EQ(instances[i]->get_field4_if_field2(), expected);
  }
}

TYPED_TEST_P(BasicClassTest, SetFieldsAfterNew) {
  TypeParam::initialize_storage();
  TypeParam** instances = new TypeParam*[kTestSize];

  // Create a few instances
  for (int i = 0; i < kTestSize; ++i) {
    instances[i] = new TypeParam();

    instances[i]->field0 = i;
    EXPECT_EQ(instances[i]->field0, i);

    instances[i]->field4 = i*i;
    EXPECT_EQ(instances[i]->field4, i*i);

    instances[i]->field2 = i % 3 == 0;
    EXPECT_EQ(instances[i]->field2, i % 3 == 0);

    int expected = i % 3 == 0 ? i*i : -1;
    EXPECT_EQ(instances[i]->get_field4_if_field2(), expected);
  }

  // Do increment.
  for (int i = 0; i < kTestSize; ++i) {
    instances[i]->add_field0_to_field4();
  }

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    int expected = i % 3 == 0 ? i + i*i : -1;
    EXPECT_EQ(instances[i]->get_field4_if_field2(), expected);
  }
}

TYPED_TEST_P(BasicClassTest, Constructor) {
  TypeParam::initialize_storage();
  TypeParam** instances = new TypeParam*[kTestSize];

  // Create a few instances
  for (int i = 0; i < kTestSize; ++i) {
    instances[i] = new TypeParam(3*i, 5*i);
  }

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    EXPECT_EQ((int) instances[i]->field0, 3*i);
    EXPECT_EQ((int) instances[i]->field4, 5*i);
  }
}

TYPED_TEST_P(BasicClassTest, PlacementNew) {
  TypeParam::initialize_storage();
  EXPECT_EQ(TypeParam::size(), 0UL);

  TypeParam* obj = new(TypeParam::get_uninitialized(4)) TypeParam();
  EXPECT_EQ(obj->id(), 4);
}

REGISTER_TYPED_TEST_CASE_P(BasicClassTest,
                           Fields,
                           SetFieldsAfterNew,
                           Constructor,
                           PlacementNew);

INSTANTIATE_TYPED_TEST_CASE_P(Valid, BasicClassTest, TestClassV);
INSTANTIATE_TYPED_TEST_CASE_P(Zero, BasicClassTest, TestClassZ_Soa);
INSTANTIATE_TYPED_TEST_CASE_P(ZeroAos, BasicClassTest, TestClassZ_Aos);

}  // namespace
