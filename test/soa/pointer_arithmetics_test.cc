#include "gtest/gtest.h"
#include "soa/soa.h"

namespace {
using ikra::soa::SoaLayout;

static const int kClassMaxInst = 1024;
static const uint32_t kTestSize = 40;

// Pointer arithmetics works only in valid addressing mode.
class TestClass : public SoaLayout<TestClass, kClassMaxInst, sizeof(int)> {
 public:
  IKRA_INITIALIZE_CLASS

  int_ field0;
};

IKRA_HOST_STORAGE(TestClass)


TEST(PointerArithmeticsTest, IncrementDecrement) {
  TestClass::initialize_storage();

  auto first = new TestClass[kTestSize];
  EXPECT_EQ(TestClass::storage().size(), kTestSize);

  int counter = 0;
  for (auto it = first; it < first + kTestSize; ++it, counter += 2) {
    it->field0 = counter;
  }

  for (int i = 0; i < static_cast<int>(kTestSize); ++i) {
    EXPECT_EQ(TestClass::get(i)->field0, 2*i);
  }
}

}  // namespace
