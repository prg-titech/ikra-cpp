#include "gtest/gtest.h"

#include "executor/cuda_executor.h"
#include "soa/soa.h"

using ikra::soa::SoaLayout;
using ikra::executor::cuda::construct;

const static int kTestSize = 12;

__device__ char data_buffer[10000];


class Vertex : public SoaLayout<Vertex, 1000> {
 public:
  IKRA_INITIALIZE_CLASS(data_buffer)

  __device__ Vertex(int f0, int f1) : field0(f0), field1(f1) {}

  int_ field0;
  int_ field1;

  __device__ void add_fields(int increment) {
    field0 = field0 + field1 + increment + this->id();

    Vertex::get(0)->field0.get();
  }
};

// Cannot run "cuda_execute" inside gtest case.
void run_test() {
  Vertex::cuda_initialize_storage();

  Vertex* first = construct<Vertex>(kTestSize, 5, 6);
  cuda_execute(Vertex, add_fields, kTestSize, first, 10)

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    int actual = Vertex::get_uninitialized(i)->field0;
    int expected = 10 + 5 + 6 + i;
    EXPECT_EQ(actual, expected);
  }

  // Make sure that we had no CUDA failures.
  gpuErrchk(cudaPeekAtLastError());
}

TEST(MinimumCudaTest, ConstructAndExecute) {
  run_test();
}
