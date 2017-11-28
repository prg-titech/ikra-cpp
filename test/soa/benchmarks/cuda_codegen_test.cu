#include <stdio.h>

#define NDEBUG    // No asserts.

#include "executor/cuda_executor.h"
#include "soa/soa.h"

using ikra::soa::SoaLayout;
using ikra::executor::cuda::construct;

const static int kTestSize = 12;

class TestClass : public SoaLayout<TestClass, 1000> {
 public:
  IKRA_INITIALIZE_CLASS

  __device__ TestClass(int f0, int f1) : field0(f0), field1(f1) {}

  int_(field0);
  int_(field1);

  __device__ void add_fields(int increment) {
    field0 = field0 + field1 + increment + this->id() + 0x7777;

    TestClass::get(0)->field0.get();
  }
};

IKRA_DEVICE_STORAGE(TestClass)


int main() {
  TestClass::initialize_storage();

  TestClass* first = construct<TestClass>(kTestSize, 0x6666, 0x5555);
  cuda_execute(&TestClass::add_fields, 0x8888);

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    int actual = TestClass::get_uninitialized(i)->field0;
    int expected = 0x7777 + 0x6666 + 0x5555 + i + 0x8888;

    if (actual == expected) {
      printf("[OK] ");
    } else {
      printf("[FAIL]\n");
      return 1;
    }
  }

  printf("\n");

  // Make sure that we had no CUDA failures.
  gpuErrchk(cudaPeekAtLastError());
  return 0;
}

// Extra methods for code isolation: Will show up in .S file.
__device__ int int_result;

__global__ void write_field0(TestClass* instance) {
  instance->field0 = 0x7777;
}

__global__ void read_field0(TestClass* instance) {
  int_result = instance->field0;
}

__global__ void write_field1(TestClass* instance) {
  instance->field1 = 0x7777;
}

__global__ void read_field1(TestClass* instance) {
  int_result = instance->field1;
}

// Compare with explicit, hand-written SOA code.
__device__ int explicit_field0[100];

__global__ void explicit_write_field0(uintptr_t id) {
  explicit_field0[id] = 0x7777;
}

__global__ void explicit_read_field0(uintptr_t id) {
  int_result = explicit_field0[id];
}
