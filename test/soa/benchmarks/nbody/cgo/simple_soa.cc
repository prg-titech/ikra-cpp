// N-Body Simulation
// Code adapted from: http://physics.princeton.edu/~fpretori/Nbody/code.htm

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "benchmark.h"

using namespace std;
using IdType = uintptr_t;

static const int kIterations = 50000000;
static const double kMaxMass = 1000;
static const double kTimeInterval = 0.5;

static const double kGravityConstant = 6.673e-11;   // gravitational constant

double a_Body_mass[kNumBodies];
double a_Body_position_0[kNumBodies];
double a_Body_position_1[kNumBodies];
double a_Body_velocity_0[kNumBodies];
double a_Body_velocity_1[kNumBodies];
double a_Body_force_0[kNumBodies];
double a_Body_force_1[kNumBodies];

#define RAND (1.0 * rand() / RAND_MAX)

void Body_initialize(IdType i, double mass, double pos_x, double pos_y,
                     double vel_x, double vel_y) {
  a_Body_position_0[i] = pos_x;
  a_Body_position_1[i] = pos_y;
  a_Body_velocity_0[i] = vel_x;
  a_Body_velocity_1[i] = vel_y;
}

void Body_codegen_simple_update(IdType self, double dt) {
  a_Body_position_0[self] += a_Body_velocity_0[self]*dt;
  a_Body_position_1[self] += a_Body_velocity_1[self]*dt;
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
}

void run_simple() {
  int iterations = kIterations * getR();
  for (int i = 0; i < iterations; ++i) {
    for (IdType j = 0; j < kNumBodies; ++j) {
      Body_codegen_simple_update(j, kTimeInterval);
    }
  }
}

int main() {
  instantiation();
  // Average R runs. 
  double time_simple = measure<>::execution(run_simple);
  time_simple /= getR();
  printf("%f\n", time_simple);

  return 0;
}
