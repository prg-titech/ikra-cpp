// N-Body Simulation
// Code adapted from: http://physics.princeton.edu/~fpretori/Nbody/code.htm

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "benchmark.h"
#include "executor/cuda_executor.h"

using namespace std;
using IdType = uintptr_t;

static const int kIterations = 5;
static const int kNumBodies = 8000;
static const double kMaxMass = 1000;
static const double kTimeInterval = 0.5;

static const double kGravityConstant = 6.673e-11;   // gravitational constant

struct Container {
  double a_Body_mass[kNumBodies];
  double a_Body_position_0[kNumBodies];
  double a_Body_position_1[kNumBodies];
  double a_Body_velocity_0[kNumBodies];
  double a_Body_velocity_1[kNumBodies];
  double a_Body_force_0[kNumBodies];
  double a_Body_force_1[kNumBodies];
};

__device__ Container d_container;
Container h_container;

#define RAND (1.0 * rand() / RAND_MAX)

void Body_initialize(IdType i, double mass, double pos_x, double pos_y,
                     double vel_x, double vel_y) {
  h_container.a_Body_mass[i] = mass;
  h_container.a_Body_position_0[i] = pos_x;
  h_container.a_Body_position_1[i] = pos_y;
  h_container.a_Body_velocity_0[i] = vel_x;
  h_container.a_Body_velocity_1[i] = vel_y;
  h_container.a_Body_force_0[i] = 0.0;
  h_container.a_Body_force_1[i] = 0.0;
}

__device__ void Body_add_force(IdType self, IdType body) {
  if (self == body) return;
  double EPS = 0.01;    // Softening parameter (just to avoid infinities).
  double dx = d_container.a_Body_position_0[body]
              - d_container.a_Body_position_0[self];
  double dy = d_container.a_Body_position_1[body]
              - d_container.a_Body_position_1[self];
  double dist = sqrt(dx*dx + dy*dy);
  double F = kGravityConstant
             * d_container.a_Body_mass[self]*d_container.a_Body_mass[body]
             / (dist*dist + EPS*EPS);
  d_container.a_Body_force_0[self] += F*dx / dist;
  d_container.a_Body_force_1[self] += F*dy / dist;
}

__device__  void Body_add_all_forces_to_this(IdType self) {
  for (IdType i = 0; i < kNumBodies; ++i) {
    Body_add_force(self, i);
  }
}

__global__ void kernel_Body_add_all_forces_to_this() {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < kNumBodies) {
    Body_add_all_forces_to_this(tid);
  }
}

__device__ void Body_reset_force(IdType self) {
  d_container.a_Body_force_0[self] = 0.0;
  d_container.a_Body_force_1[self] = 0.0;
}

__global__ void kernel_Body_reset_force() {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < kNumBodies) {
    Body_reset_force(tid);
  }
}

__device__ void Body_update(IdType self, double dt) {
  d_container.a_Body_velocity_0[self] +=
      d_container.a_Body_force_0[self]*dt / d_container.a_Body_mass[self];
  d_container.a_Body_velocity_1[self] +=
      d_container.a_Body_force_1[self]*dt / d_container.a_Body_mass[self];
  d_container.a_Body_position_0[self] +=
      d_container.a_Body_velocity_0[self]*dt;
  d_container.a_Body_position_1[self] +=
      d_container.a_Body_velocity_1[self]*dt;

  if (d_container.a_Body_position_0[self] < -1
        || d_container.a_Body_position_0[self] > 1) {
    d_container.a_Body_velocity_0[self] = -d_container.a_Body_velocity_0[self];
  }
  if (d_container.a_Body_position_1[self] < -1
        || d_container.a_Body_position_1[self] > 1) {
    d_container.a_Body_velocity_1[self] = -d_container.a_Body_velocity_1[self];
  }
}

__global__ void kernel_Body_update(double dt) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < kNumBodies) {
    Body_update(tid, dt);
  }
}

__device__ void Body_codegen_simple_update(IdType self, double dt) {
  for (int i = 0; i < 1000; ++i) {
    d_container.a_Body_velocity_0[self] +=
        d_container.a_Body_force_0[self]*dt / d_container.a_Body_mass[self];
    d_container.a_Body_velocity_1[self] +=
        d_container.a_Body_force_1[self]*dt / d_container.a_Body_mass[self];
    d_container.a_Body_position_0[self] +=
        d_container.a_Body_velocity_0[self]*dt;
    d_container.a_Body_position_1[self] +=
        d_container.a_Body_velocity_1[self]*dt;
  }
}

__global__ void kernel_Body_codegen_simple_update(double dt) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < kNumBodies) {
    Body_codegen_simple_update(tid, dt);
  }
}

void instantiation() {
  srand(42);

  // Create objects.
  for (IdType i = 0; i < kNumBodies; ++i) {
    double mass = (RAND/2 + 0.5) * kMaxMass;
    double pos_x = RAND*2 - 1;
    double pos_y = RAND*2 - 1;
    double vel_x = (RAND - 0.5) / 1000;
    double vel_y = (RAND - 0.5) / 1000;
    Body_initialize(i, mass, pos_x, pos_y, vel_x, vel_y);
  }

  // Transfer data to GPU.
  cudaMemcpyToSymbol(d_container, &h_container, sizeof(Container), 0,
                     cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  assert(cudaPeekAtLastError() == cudaSuccess);
}

void run_simulation() {
  uintptr_t num_blocks = ikra::executor::cuda::cuda_blocks_1d(kNumBodies);
  uintptr_t num_threads = ikra::executor::cuda::cuda_threads_1d(kNumBodies);

  for (int i = 0; i < kIterations; ++i) {
    // Reset forces.
    kernel_Body_reset_force<<<num_blocks, num_threads>>>();

    // Update forces.
    kernel_Body_add_all_forces_to_this<<<num_blocks, num_threads>>>();

    // Update velocities and positions.
    kernel_Body_update<<<num_blocks, num_threads>>>(kTimeInterval);
  }

  cudaDeviceSynchronize();
  assert(cudaPeekAtLastError() == cudaSuccess);
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
  uint64_t time_instantiation = measure<>::execution(instantiation);
  gpuErrchk(cudaPeekAtLastError());

  uint64_t time_simulation = measure<>::execution(run_simulation);
  gpuErrchk(cudaPeekAtLastError());

  uint64_t time_simple = measure<>::execution(run_simple);
  gpuErrchk(cudaPeekAtLastError());

  // Transfer data to CPU.
  cudaMemcpyFromSymbol(&h_container, d_container, sizeof(Container), 0,
                       cudaMemcpyDeviceToHost);
  gpuErrchk(cudaPeekAtLastError());

  // Calculate checksum
  int checksum = 11;
  for (uintptr_t i = 0; i < kNumBodies; i++) {
    checksum += reinterpret_cast<int>(
        r_float2int(h_container.a_Body_position_0[i]));
    checksum += reinterpret_cast<int>(
        r_float2int(h_container.a_Body_position_1[i]));
    checksum = checksum % 1234567;

    if (i < 10) {
      printf("VALUE[%lu] = %f, %f\n", i,
             h_container.a_Body_position_0[i],
             h_container.a_Body_position_1[i]);
    }
  }

  printf("instantiation: %lu\nsimulation: %lu\nsimple: %lu\nchecksum: %i\n",
         time_instantiation, time_simulation, time_simple, checksum);
  return 0;
}

__global__  void codengen_simple_update(IdType body, double dt) {
  Body_codegen_simple_update(body, dt);
}
