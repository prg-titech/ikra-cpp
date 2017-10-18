// This is a minimal test case whose purpose is to check if the compiler
// generates efficient assembly code. Should be built with -O3 optimization
// and not through Bazel (is not set up to generate such optimized code).

#include <stdio.h>

#define NDEBUG    // No asserts.
#include "../../../soa/soa.h"

using ikra::soa::SoaLayout;
using ikra::soa::kAddressModeZero;
using ikra::soa::kAddressModeValid;

static const uint32_t kClassMaxInst = 0x1234;

class TestClass : public SoaLayout<TestClass, 17,
                                   kClassMaxInst, kAddressModeZero> {
 public:
  static Storage storage;

  int_<0> field0;
  int_<4> field1;

  void increase_field0() {
    field0 *= 0x5555;
  }

  void increase_field1() {
    field1 *= 0x4444;
  }
};

TestClass::Storage TestClass::storage;


int main() {
  TestClass* instance = new TestClass();
  instance->field0 = 0x7777;
  instance->field1 = 0x8888;
  instance->increase_field0();
  instance->increase_field1();

  // Expected output: FIELD0: 61166, FIELD1: 69904
  printf("FIELD0: %i, FIELD1: %i\n", (int) instance->field0,
                                     (int) instance->field1);
}
