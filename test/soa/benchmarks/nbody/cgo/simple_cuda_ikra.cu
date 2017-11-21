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

class Body : public SoaLayout<Body, kNumBodies> {
 public:
  IKRA_INITIALIZE_CLASS

  Body(double mass, double pos_x, double pos_y,
       double vel_x, double vel_y) : mass_(mass) {
    position_[0] = pos_x;
    position_[1] = pos_y;
    velocity_[0] = vel_x;
    velocity_[1] = vel_y;
  }

  double_ mass_;
  array_(double, 2) position_;
  array_(double, 2) velocity_;
  array_(double, 2) force_;

  __device__ void codengen_simple_update(double dt) {
    position_[0] += velocity_[0]*dt;
    position_[1] += velocity_[1]*dt;
  }
};

IKRA_DEVICE_STORAGE(Body)

__global__ void kernel_instantiate(int num_instances) {
  Body::storage().increase_size(num_instances);
}

void instantiation_gpu() {
  kernel_instantiate<<<1, 1>>>(kNumBodies);
  cudaDeviceSynchronize();
}


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
    new Body(mass, pos_x, pos_y, vel_x, vel_y);
  }
  cudaDeviceSynchronize();
}

void run_simple() {
  for (int i = 0; i < kIterations*100; ++i) {
    // In cuda_execute_fixed_size, the number of objects is a compile-time
    // constant. Similar to the SOA and AOS baseline benchmarks.
    // Alternatively, we could also use cuda_execute here, but the generated
    // assembly code will be slightly less efficient.
    cuda_execute_fixed_size(&Body::codengen_simple_update,
                            kNumBodies, kTimeInterval);
  }
}

int main() {
  instantiation_gpu();
  gpuErrchk(cudaPeekAtLastError());

  // Average R runs. 
  double time_simple = measure<>::execution(run_simple);
  printf("%f\n", time_simple);

  return 0;
}

int main2() {
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    // Initialize object storage.
  Body::initialize_storage();

  uint64_t time_instantiation = measure<>::execution(instantiation);
  gpuErrchk(cudaPeekAtLastError());

  uint64_t time_simple = measure<>::execution(run_simple);
  gpuErrchk(cudaPeekAtLastError());

  // Calculate checksum
  int checksum = 11;
  for (uintptr_t i = 0; i < kNumBodies; i++) {
    checksum += reinterpret_cast<int>(
        r_float2int(Body::get(i)->position_[0]));
    checksum += reinterpret_cast<int>(
        r_float2int(Body::get(i)->position_[1]));
    checksum = checksum % 1234567;

    if (i < 10) {
      printf("VALUE[%lu] = %f, %f\n", i,
             (double) Body::get(i)->position_[0],
             (double) Body::get(i)->position_[1]);
    }
  }

  printf("instantiation: %lu\nsimple: %lu\n checksum: %i\n",
         time_instantiation, time_simple, checksum);
  return 0;
}
