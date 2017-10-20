#include "gtest/gtest.h"
#include "../../soa/soa.h"

namespace {
using ikra::soa::SoaLayout;

static const int kClassMaxInst = 1024;
static const int kTestSize = 40;

// Pointer arithmetics works only in valid addressing mode.
class TestClass : public SoaLayout<TestClass, 4,
                                   kClassMaxInst, sizeof(int)> {
 public:
  static Storage storage;

  int_<0> field0;
};

TestClass::Storage TestClass::storage;

TEST(PointerArithmeticsTest, IncrementDecrement) {
  auto first = new TestClass[kTestSize];
  EXPECT_EQ(TestClass::storage.size, kTestSize);

  int counter = 0;
  for (auto it = first; it < first + kTestSize; ++it, counter += 2) {
    it->field0 = counter;
  }

  for (int i = 0; i < kTestSize; ++i) {
    EXPECT_EQ(TestClass::get(i)->field0, 2*i);
  }
}

}  // namespace
