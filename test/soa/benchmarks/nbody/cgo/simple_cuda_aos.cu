// N-Body Simulation
// Code adapted from: http://physics.princeton.edu/~fpretori/Nbody/code.htm

#define NDEBUG    // No asserts.

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "benchmark.h"
#include "executor/cuda_executor.h"
#include "executor/executor.h"
#include "soa/soa.h"

using namespace std;
using ikra::executor::execute;
using ikra::executor::Iterator;
using ikra::soa::SoaLayout;

static const int kIterations = 50;
static const double kMaxMass = 1000;
static const double kTimeInterval = 0.5;

#define RAND (1.0 * rand() / RAND_MAX)

class Body {
 public:
  __host__ Body(double mass, double pos_x, double pos_y,
       double vel_x, double vel_y) {
    position_[0] = pos_x;
    position_[1] = pos_y;
    velocity_[0] = vel_x;
    velocity_[1] = vel_y;
  }

  double position_[2];
  double velocity_[2];

  // Uncomment for AOS-32
  // (=32 additional floats, equal to 16 additional doubles)
  // double force_[16];

  __device__ void codengen_simple_update(double dt) {
    position_[0] += velocity_[0]*dt;
    position_[1] += velocity_[1]*dt;
  }
};

char h_Body_arena[kNumBodies * sizeof(Body)];
__device__ char d_Body_arena[kNumBodies * sizeof(Body)];
char* Body_arena_head = h_Body_arena;

// TODO: Benchmark instance creation on GPU.
void instantiation() {
  srand(42);

  // Create objects.
  for (int i = 0; i < kNumBodies; ++i) {
    double mass = (RAND/2 + 0.5) * kMaxMass;
    double pos_x = RAND*2 - 1;
    double pos_y = RAND*2 - 1;
    double vel_x = (RAND - 0.5) / 1000;
    double vel_y = (RAND - 0.5) / 1000;
    new(Body_arena_head) Body(mass, pos_x, pos_y, vel_x, vel_y);
    Body_arena_head += sizeof(Body);
  }

  cudaMemcpyToSymbol(d_Body_arena, &h_Body_arena, sizeof(h_Body_arena), 0,
                     cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}

__global__ void kernel_Body_codegen_simple_update(double dt) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < kNumBodies) {
    Body* body = reinterpret_cast<Body*>(d_Body_arena + sizeof(Body)*tid);
    body->codengen_simple_update(dt);
  }
}

void run_simple() {
  uintptr_t num_blocks = ikra::executor::cuda::cuda_blocks_1d(kNumBodies);
  uintptr_t num_threads = ikra::executor::cuda::cuda_threads_1d(kNumBodies);

  for (int i = 0; i < kIterations*100; ++i) {
    kernel_Body_codegen_simple_update<<<num_blocks, num_threads>>>(
        kTimeInterval);
  }

  cudaDeviceSynchronize();
  assert(cudaPeekAtLastError() == cudaSuccess);
}

int main() {
  instantiation();
  double time_simple = measure<>::execution(run_simple);
  printf("%f\n", time_simple);

  return 0;
}

int main2() {
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  uint64_t time_instantiation = measure<>::execution(instantiation);
  gpuErrchk(cudaPeekAtLastError());

  uint64_t time_simple = measure<>::execution(run_simple);
  gpuErrchk(cudaPeekAtLastError());

  exit(0);

  cudaMemcpyFromSymbol(&h_Body_arena, d_Body_arena, sizeof(h_Body_arena), 0,
                       cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Calculate checksum
  int checksum = 11;
  for (uintptr_t i = 0; i < kNumBodies; i++) {
    Body* body = reinterpret_cast<Body*>(h_Body_arena + sizeof(Body)*i);
    checksum += reinterpret_cast<int>(
        r_float2int(body->position_[0]));
    checksum += reinterpret_cast<int>(
        r_float2int(body->position_[1]));
    checksum = checksum % 1234567;

    if (i < 10) {
      printf("VALUE[%lu] = %f, %f\n", i,
             (double) body->position_[0],
             (double) body->position_[1]);
    }
  }

  printf("instantiation: %lu\nsimple: %lu\n checksum: %i\n",
         time_instantiation, time_simple, checksum);
  return 0;
}
