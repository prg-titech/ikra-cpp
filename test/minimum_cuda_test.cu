#include <array>

#include "gtest/gtest.h"

#include "executor/cuda_executor.h"
#include "soa/soa.h"

using ikra::soa::IndexType;
using ikra::soa::SoaLayout;
using ikra::executor::cuda::construct;

const static int kTestSize = 12;

class Vertex : public SoaLayout<Vertex, 1000> {
 public:
  IKRA_INITIALIZE_CLASS

  __host__ __device__ Vertex(int f0, int f1) : field0(f0), field1(f1) {}

  int_ field0;
  int_ field1;

  __device__ void add_fields(int increment) {
    field0 = field0 + field1 + increment + this->id();
  }
};

IKRA_DEVICE_STORAGE(Vertex);


// Cannot run "cuda_execute" inside gtest case.
void run_test_construct_and_execute() {
  Vertex::initialize_storage();
  EXPECT_EQ(Vertex::size(), 0UL);

  Vertex* first = construct<Vertex>(kTestSize, 5, 6);
  gpuErrchk(cudaPeekAtLastError());

  cuda_execute(&Vertex::add_fields, first, 12, 10);

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    int actual = Vertex::get(i)->field0;
    int expected = 10 + 5 + 6 + i;
    EXPECT_EQ(actual, expected);
  }

  // Copy size to host memory and compare.
  EXPECT_EQ(Vertex::size(), static_cast<IndexType>(kTestSize));

  // Make sure that we had no CUDA failures.
  gpuErrchk(cudaPeekAtLastError());
}

void run_test_host_side_assignment() {
  Vertex::initialize_storage();
  EXPECT_EQ(Vertex::size(), 0UL);

  Vertex* first = construct<Vertex>(kTestSize, 5, 6);
  cuda_execute(&Vertex::add_fields, first, kTestSize, 10);

  for (int i = 0; i < kTestSize; ++i) {
    Vertex::get(i)->field0 = Vertex::get(i)->field0*Vertex::get(i)->field0;
  }

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    int actual = Vertex::get(i)->field0;
    int expected = (10 + 5 + 6 + i)*(10 + 5 + 6 + i);
    EXPECT_EQ(actual, expected);
  }

  // Copy size to host memory and compare.
  EXPECT_EQ(Vertex::size(), static_cast<IndexType>(kTestSize));

  // Make sure that we had no CUDA failures.
  gpuErrchk(cudaPeekAtLastError());
}

void run_test_host_side_new() {
  Vertex::initialize_storage();
  EXPECT_EQ(Vertex::size(), 0UL);

  std::array<Vertex*, kTestSize> vertices;

  for (int i = 0; i < kTestSize; ++i) {
    vertices[i] = new Vertex(i + 1, i * i);
    EXPECT_EQ(vertices[i]->id(), static_cast<IndexType>(i));
  }

  cuda_execute(&Vertex::add_fields, vertices[0], kTestSize, 10);

  // Check result.
  for (int i = 0; i < kTestSize; ++i) {
    int actual = Vertex::get(i)->field0;
    int expected = 10 + i + (i + 1) + (i*i);
    EXPECT_EQ(actual, expected);

    actual = vertices[i]->field1;
    expected = i*i;
    EXPECT_EQ(actual, expected);
  }

  // Copy size to host memory and compare.
  EXPECT_EQ(Vertex::size(), static_cast<IndexType>(kTestSize));

  // Make sure that we had no CUDA failures.
  gpuErrchk(cudaPeekAtLastError());
}

TEST(MinimumCudaTest, ConstructAndExecute) {
  run_test_construct_and_execute();
}

TEST(MinimumCudaTest, HostSideAssignment) {
  run_test_host_side_assignment();
}

TEST(MinimumCudaTest, HostSideNew) {
  run_test_host_side_new();
}
