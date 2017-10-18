#include "gtest/gtest.h"
#include "../../soa/soa.h"

namespace {
using ikra::soa::SoaLayout;

static const uint32_t kClassMaxInst = 1024;
static const uint32_t kTestSize = 40;

class TestClass : public SoaLayout<TestClass, 17, kClassMaxInst> {
 public:
  static Storage storage;

  int_<0> field0;
  double_<4> field1;
  bool_<12> field2;
  int_<13> field4;

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

TestClass::Storage TestClass::storage;

TEST(SimpleClassTest, Fields) {
  TestClass** instances = new TestClass*[kTestSize];

  // Create a few instances
  for (int i = 0; i < kTestSize; ++i) {
    instances[i] = new TestClass();
  }

  // Set values.
  for (int i = 0; i < kTestSize; ++i) {
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

}  // namespace
