// N-Body Simulation
// Code adapted from: http://physics.princeton.edu/~fpretori/Nbody/code.htm

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <array>

#include "benchmark.h"

using namespace std;

static const int kIterations = 5;
static const int kNumBodies = 8000;
static const double kMaxMass = 1000;
static const double kTimeInterval = 0.5;

static const double kGravityConstant = 6.673e-11;   // gravitational constant

#define RAND (1.0 * rand() / RAND_MAX)

char Body_arena[kNumBodies*8*8];
char* Body_arena_head = Body_arena;

class Body {
 public:
  Body(double mass, double pos_x, double pos_y, double vel_x, double vel_y)
      : mass_(mass) {
    position_[0] = pos_x;
    position_[1] = pos_y;
    velocity_[0] = vel_x;
    velocity_[1] = vel_y;
    this->reset_force();
  }

  double mass_;
  std::array<double, 2> position_;
  std::array<double, 2> velocity_;
  std::array<double, 2> force_;

  void add_force(Body* body) {
    if (this == body) return;

    double EPS = 0.01;    // Softening parameter (just to avoid infinities).
    double dx = body->position_[0] - position_[0];
    double dy = body->position_[1] - position_[1];
    double dist = sqrt(dx*dx + dy*dy);
    double F = kGravityConstant*mass_*body->mass_ / (dist*dist + EPS*EPS);
    force_[0] += F*dx / dist;
    force_[1] += F*dy / dist;
  }

  void add_force_to_all() {
    for (int i = 0; i < kNumBodies; ++i) {
      Body* body = reinterpret_cast<Body*>(Body_arena + sizeof(Body)*i);
      body->add_force(this);
    }
  }

  void reset_force() {
    force_[0] = 0.0;
    force_[1] = 0.0;
  }

  void update(double dt) {
    velocity_[0] += force_[0]*dt / mass_;
    velocity_[1] += force_[1]*dt / mass_;
    position_[0] += velocity_[0]*dt;
    position_[1] += velocity_[1]*dt;

    if (position_[0] < -1 || position_[0] > 1) {
      velocity_[0] = -velocity_[0];
    }
    if (position_[1] < -1 || position_[1] > 1) {
      velocity_[1] = -velocity_[1];
    }
  }

  void codengen_simple_update(double dt) {
    for (int i = 0; i < 100; ++i) {
      velocity_[0] += force_[0]*dt / mass_;
      velocity_[1] += force_[1]*dt / mass_;
      position_[0] += velocity_[0]*dt;
      position_[1] += velocity_[1]*dt;
    }
  }
};


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
}

void run_simulation() {
  for (int i = 0; i < kIterations; ++i) {
    // Reset forces.
    for (int j = 0; j < kNumBodies; ++j) {
      Body* body = reinterpret_cast<Body*>(Body_arena + sizeof(Body)*j);
      body->reset_force();
    }

    // Update forces.
    for (int j = 0; j < kNumBodies; ++j) {
      Body* body = reinterpret_cast<Body*>(Body_arena + sizeof(Body)*j);
      body->add_force_to_all();
    }

    // Update velocities and positions.
    for (int j = 0; j < kNumBodies; ++j) {
      Body* body = reinterpret_cast<Body*>(Body_arena + sizeof(Body)*j);
      body->update(kTimeInterval);
    }
  }
}

void run_simple() {
  for (int i = 0; i < kIterations*100; ++i) {
    for (int j = 0; j < kNumBodies; ++j) {
      Body* body = reinterpret_cast<Body*>(Body_arena + sizeof(Body)*j);
      body->codengen_simple_update(kTimeInterval);
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
    checksum += reinterpret_cast<int>(r_float2int(
        reinterpret_cast<Body*>(Body_arena + sizeof(Body)*i)->position_[0]));
    checksum += reinterpret_cast<int>(r_float2int(
        reinterpret_cast<Body*>(Body_arena + sizeof(Body)*i)->position_[1]));
    checksum = checksum % 1234567;

    if (i < 10) {
      printf("VALUE[%lu] = %f, %f\n", i,
          reinterpret_cast<Body*>(Body_arena + sizeof(Body)*i)->position_[0],
          reinterpret_cast<Body*>(Body_arena + sizeof(Body)*i)->position_[1]);
    }
  }

  printf("instantiation: %lu\nsimulation: %lu\nsimple: %lu\n checksum: %i\n",
         time_instantiation, time_simulation, time_simple, checksum);
  return 0;
}
