// This is a minimal test case whose purpose is to check if the compiler
// generates efficient assembly code. Should be built with -O3 optimization
// and not through Bazel (is not set up to generate such optimized code).

#include <stdio.h>

#define NDEBUG    // No asserts.
#include "soa/soa.h"

using ikra::soa::SoaLayout;
using ikra::soa::kAddressModeZero;

// Compiler flag determines addressing mode.
#ifdef ADDR_ZERO
#define ADDR_MODE kAddressModeZero
#else
#define ADDR_MODE 4
#endif

static const uint32_t kClassMaxInst = 0x1234;

class TestClass : public SoaLayout<TestClass, kClassMaxInst, ADDR_MODE> {
 public:
  #include IKRA_INITIALIZE_CLASS

  int___ field0;
  int___ field1;

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

  // Expected output: FIELD0: 668085635, FIELD1: 610821152
  printf("FIELD0: %i, FIELD1: %i\n", (int) instance->field0,
                                     (int) instance->field1);

  // Return 0 if correct result.
  return !(instance->field0 == 668085635 && instance->field1 == 610821152);
}

// Extra methods for code isolation: Will show up in .S file.
TestClass* new_instance() {
  return new TestClass();
}

void write_field0(TestClass* instance) {
  instance->field0 = 0x7777;
}

int read_field0(TestClass* instance) {
  return instance->field0;
}

// Compare with explicit, hand-written SOA code.
int* explicit_field0;

void explicit_write_field0(uintptr_t id) {
  explicit_field0[id] = 0x7777;
}

int explicit_read_field0(uintptr_t id) {
  return explicit_field0[id];
}
