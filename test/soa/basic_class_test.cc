#include "gtest/gtest.h"
#include "../../soa/soa.h"

namespace {
using ikra::soa::SoaLayout;
using ikra::soa::kAddressModeZero;

static const int kClassMaxInst = 1024;
static const int kTestSize = 40;

// Note: All the extra keywords (typename etc.) are required because this
// class is templatized.
template<int AddressMode>
class TestClass : public SoaLayout<TestClass<AddressMode>, 17,
                                   kClassMaxInst, AddressMode> {
 public:
  using SelfSuper = SoaLayout<TestClass<AddressMode>, 17,
                              kClassMaxInst, AddressMode>;

  static typename SelfSuper::Storage storage;

  //typename SelfSuper::template int_<0> field0;
  typename SelfSuper::template int_<0> field0;
  typename SelfSuper::template double_<4> field1;
  typename SelfSuper::template bool_<12> field2;
  typename SelfSuper::template int_<13> field4;

  TestClass() {}

  TestClass(int field0_a, int field4_a) : field4(field4_a) {
    field0 = field0_a;
  }

  void add_field0_to_field4() {
    field4 = field4 + field0;
  }

  int get_field4_if_field2() {
    if (field2) {
      return field4;
    } else {
      return -1;
    }
  }
};

template<>
TestClass<sizeof(int)>::Storage TestClass<sizeof(int)>::storage;

template<>
TestClass<kAddressModeZero>::Storage TestClass<kAddressModeZero>::storage;

template<typename T>
class BasicClassTest : public testing::Test {};

TYPED_TEST_CASE_P(BasicClassTest);

TYPED_TEST_P(BasicClassTest, Fields) {
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

REGISTER_TYPED_TEST_CASE_P(BasicClassTest,
                           Fields,
                           SetFieldsAfterNew,
                           Constructor);

INSTANTIATE_TYPED_TEST_CASE_P(Valid, BasicClassTest,
                              TestClass<sizeof(int)>);
INSTANTIATE_TYPED_TEST_CASE_P(Zero, BasicClassTest,
                              TestClass<kAddressModeZero>);

}  // namespace
