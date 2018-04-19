#include "executor/executor.h"
#include "soa/soa.h"

#include "benchmark.h"
#include "configuration.h"

using ikra::soa::IndexType;
using ikra::soa::SoaLayout;
using ikra::executor::construct;
using ikra::executor::execute;

class DummyClass : public SoaLayout<DummyClass, NUM_INST> {
 public:
  IKRA_INITIALIZE_CLASS

  DummyClass(int f0, int f2): field0(f0), field2(f2) {
    for (int i = 0; i < ARRAY_SIZE; ++i) {
      field1[i] = id()*17 + i;
    }
  }

  int_ field0;

  array_(int, ARRAY_SIZE, object) field1;

  int_ field2;

  void update_field1(int increment) {
    for (int i = 0; i < ARRAY_SIZE; ++i) {
      field1[i] += increment + field0 + field2;
    }
  }
};

IKRA_HOST_STORAGE(DummyClass);


void action() {
  DummyClass::DBG_print_offsets();
  DummyClass::initialize_storage();
  DummyClass* first = construct<DummyClass>(NUM_INST, 29, 1357);

#ifndef NDEBUG
  // Check values (some samples).
  for (int k = 0; k < 100; ++k) {
    int i = rand() % NUM_INST;
    assert((int) DummyClass::get(i)->field0 == 29);
    assert((int) DummyClass::get(i)->field2 == 1357);
  }
#endif  // NDEBUG

  execute(&DummyClass::update_field1, 19);
}

void run_test_construct_and_execute() {
  uint64_t time_action = measure<>::execution(action);
  printf("[AOS-CPU] Time for action: %lu\n", time_action);

#ifndef NDEBUG
  // Check result (some samples).
  for (int k = 0; k < 100; ++k) {
    int i = rand() % NUM_INST;
    for (int j = 0; j < ARRAY_SIZE; ++j) {
      int actual1 = DummyClass::get(i)->field1[j];
      int expected1 = i*17 + j + 19 + 29 + 1357;// + 19 + 29 + 1357;
      if (actual1 != expected1) {
        printf("[AOS-CPU] Dummy[%i].field1[%i]: Expected %i, but found %i\n",
               i, j, expected1, actual1);
        exit(1);
      }
    }
  }
#endif  // NDEBUG
}

int main() {
  run_test_construct_and_execute();
}
