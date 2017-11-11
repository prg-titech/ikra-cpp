// N-Body Simulation
// Code adapted from: http://physics.princeton.edu/~fpretori/Nbody/code.htm

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "benchmark.h"

using namespace std;
using IdType = uintptr_t;

static const int kIterations = 2;
static const int kNumBodies = 9600;
static const double kMaxMass = 1000;
static const double kTimeInterval = 0.5;

static const double kGravityConstant = 6.673e-11;   // gravitational constant

static const int kWindowWidth = 1000;
static const int kWindowHeight = 1000;
static const int kMaxRect = 20;

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
  a_Body_mass[i] = mass;
  a_Body_position_0[i] = pos_x;
  a_Body_position_1[i] = pos_y;
  a_Body_velocity_0[i] = vel_x;
  a_Body_velocity_1[i] = vel_y;
  a_Body_force_0[i] = 0.0;
  a_Body_force_1[i] = 0.0;
}

void Body_add_force(IdType self, IdType body) {
  if (self == body) return;

  double EPS = 0.01;    // Softening parameter (just to avoid infinities).
  double dx = a_Body_position_0[body] - a_Body_position_0[self];
  double dy = a_Body_position_1[body] - a_Body_position_1[self];
  double dist = sqrt(dx*dx + dy*dy);
  double F = kGravityConstant*a_Body_mass[self]*a_Body_mass[body]
             / (dist*dist + EPS*EPS);
  a_Body_force_0[self] += F*dx / dist;
  a_Body_force_1[self] += F*dy / dist;
}

void Body_add_force_to_all(IdType self) {
  for (IdType i = 0; i < kNumBodies; ++i) {
    Body_add_force(i, self);
  }
}

void Body_reset_force(IdType self) {
  a_Body_force_0[self] = 0.0;
  a_Body_force_1[self] = 0.0;
}

void Body_update(IdType self, double dt) {
  a_Body_velocity_0[self] += a_Body_force_0[self]*dt / a_Body_mass[self];
  a_Body_velocity_1[self] += a_Body_force_1[self]*dt / a_Body_mass[self];
  a_Body_position_0[self] += a_Body_velocity_0[self]*dt;
  a_Body_position_1[self] += a_Body_velocity_1[self]*dt;

  if (a_Body_position_0[self] < -1 || a_Body_position_0[self] > 1) {
    a_Body_velocity_0[self] = -a_Body_velocity_0[self];
  }
  if (a_Body_position_1[self] < -1 || a_Body_position_1[self] > 1) {
    a_Body_velocity_1[self] = -a_Body_velocity_1[self];
  }
}

void Body_codegen_simple_update(IdType self, double dt) {
  a_Body_velocity_0[self] += a_Body_force_0[self]*dt / a_Body_mass[self];
  a_Body_velocity_1[self] += a_Body_force_1[self]*dt / a_Body_mass[self];
  a_Body_position_0[self] += a_Body_velocity_0[self]*dt;
  a_Body_position_1[self] += a_Body_velocity_1[self]*dt;
}

void instantiation() {
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

void run_simulation() {
  for (int i = 0; i < kIterations; ++i) {
    // Reset forces.
    for (IdType j = 0; j < kNumBodies; ++j) {
      Body_reset_force(j);
    }

    // Update forces.
    for (IdType j = 0; j < kNumBodies; ++j) {
      Body_add_force_to_all(j);
    }

    // Update velocities and positions.
    for (IdType j = 0; j < kNumBodies; ++j) {
      Body_update(j, kTimeInterval);
    }
  }
}

void run_simple() {
  for (int i = 0; i < kIterations*10000; ++i) {
    for (IdType j = 0; j < kNumBodies; ++j) {
      Body_codegen_simple_update(j, kTimeInterval);
    }
  }
}

int main() {
  uint64_t time_instantiation = measure<>::execution(instantiation);
  uint64_t time_simulation = measure<>::execution(run_simulation);
  uint64_t time_simple = measure<>::execution(run_simple);

  // Calculate checksum
  int checksum = 11;
  for (uintptr_t i = 0; i < kNumBodies; i++) {
    checksum += reinterpret_cast<int>(float_as_int(a_Body_position_0[i]));
    checksum += reinterpret_cast<int>(float_as_int(a_Body_position_1[i]));
    checksum = checksum % 1234567;
  }

  printf("instantiation: %lu\nsimulation: %lu\nsimple: %lu\n checksum: %i\n",
         time_instantiation, time_simulation, time_simple, checksum);
  return 0;
}
