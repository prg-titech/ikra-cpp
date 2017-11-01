#include <stdio.h>
#include "executor/cuda_executor.h"
#include "soa/soa.h"

using ikra::soa::SoaLayout;
using ikra::executor::cuda::construct;

const static int kTestSize = 500;

__device__ char data_buffer[10000];


class Vertex : public SoaLayout<Vertex, 1000> {
 public:
  IKRA_INITIALIZE_CLASS(data_buffer)

  __device__ Vertex(int f0, int f1) : field0(f0), field1(f1) {}

  int_ field0;
  int_ field1;

  __device__ void add_fields(int increment) {
    field0 = field0 + field1 + increment + this->id();
  }

  __device__ void foo() {
    printf("CALLING FOO on %i!\n", (int) field0);
  }
};


int main()
{
  Vertex::cuda_initialize_storage();

  Vertex* first = construct<Vertex>(kTestSize, 5, 6);
  cuda_execute(Vertex, add_fields, kTestSize, first, 10)

  // Should print: 10+5+6+i = 21+
  cuda_execute(Vertex, foo, kTestSize, first)
}
