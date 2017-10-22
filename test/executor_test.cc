#include "gtest/gtest.h"
#include "executor/array.h"
#include "executor/iterator.h"
#include "soa/soa.h"

namespace {
using ikra::soa::SoaLayout;
using ikra::executor::IteratorExecutor;
using ikra::executor::Iterator;

static const int kClassMaxInst = 1024;
static const int kTestSize = 40;

// Pointer arithmetics works only in valid addressing mode.
class TestClass : public SoaLayout<TestClass, kClassMaxInst, sizeof(int)> {
 public:
  #include IKRA_INITIALIZE_CLASS

  int___ field0;
  int___ field1;

  TestClass(int a, int b) : field0(a), field1(b) {}

  void add_field1_and_a_to_field0(int a) {
    field0 += field1 + a;
  }
};

TestClass::Storage TestClass::storage;

TEST(ExecutorTest, StdArray) {
  std::array<TestClass*, kTestSize> arr;

  for (int i = 0; i < kTestSize; ++i) {
    arr[i] = new TestClass(i + 1, 100*i + 1);
  }

  auto executor = IteratorExecutor(arr.begin(), arr.end());
  executor.execute(&TestClass::add_field1_and_a_to_field0, 50);

  // Check result
  for (int i = 0; i < kTestSize; ++i) {
    int expected = i + 1 + 100*i + 1 + 50;
    EXPECT_EQ(arr[i]->field0, expected);
  }
}

TEST(ExecutorTest, Iterator) {
  std::array<TestClass*, kTestSize> arr;

  for (int i = 0; i < kTestSize; ++i) {
    arr[i] = new TestClass(i + 1, 200*i + 1);
  }

  auto executor = IteratorExecutor(Iterator(arr[0]),
                                   ++Iterator(arr[kTestSize - 1]));
  executor.execute(&TestClass::add_field1_and_a_to_field0, 50);

  // Check result
  for (int i = 0; i < kTestSize; ++i) {
    int expected = i + 1 + 200*i + 1 + 50;
    EXPECT_EQ(arr[i]->field0, expected);
  }
}


}  // namespace

