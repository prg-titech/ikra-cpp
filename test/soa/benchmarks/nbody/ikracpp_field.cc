// N-Body Simulation
// Code adapted from: http://physics.princeton.edu/~fpretori/Nbody/code.htm

#define NDEBUG    // No asserts.

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "benchmark.h"
#include "executor/executor.h"
#include "soa/soa.h"

using namespace std;
using ikra::executor::execute;
using ikra::executor::Iterator;
using ikra::soa::SoaLayout;

static const int kIterations = 5;
static const int kNumBodies = 8000;
static const double kMaxMass = 1000;
static const double kTimeInterval = 0.5;

static const double kGravityConstant = 6.673e-11;   // gravitational constant

#define RAND (1.0 * rand() / RAND_MAX)

class Body : public SoaLayout<Body, kNumBodies> {
 public:
  IKRA_INITIALIZE_CLASS

  Body(double mass, double pos_x, double pos_y, double vel_x, double vel_y)
      : mass_(mass) {
    position0_ = pos_x;
    position1_ = pos_y;
    velocity0_ = vel_x;
    velocity1_ = vel_y;
    this->reset_force();
  }

  double_(mass_);
  double_(position0_);
  double_(position1_);
  double_(velocity0_);
  double_(velocity1_);
  double_(force0_);
  double_(force1_);

  void add_force(Body* body) {
    if (this == body) return;

    double EPS = 0.01;    // Softening parameter (just to avoid infinities).
    double dx = body->position0_ - position0_;
    double dy = body->position1_ - position1_;
    double dist = sqrt(dx*dx + dy*dy);
    double F = kGravityConstant*mass_*body->mass_ / (dist*dist + EPS*EPS);
    force0_ += F*dx / dist;
    force1_ += F*dy / dist;
  }

  void add_force_to_all() {
    execute<kNumBodies>(&Body::add_force, this);
  }

  void reset_force() {
    force0_ = 0.0;
    force1_ = 0.0;
  }

  void update(double dt) {
    velocity0_ += force0_*dt / mass_;
    velocity1_ += force1_*dt / mass_;
    position0_ += velocity0_*dt;
    position1_ += velocity1_*dt;

    if (position0_ < -1 || position0_ > 1) {
      velocity0_ = -velocity0_;
    }
    if (position1_ < -1 || position1_ > 1) {
      velocity1_ = -velocity1_;
    }
  }

  void codengen_simple_update(double dt) {
    for (int i = 0; i < 100; ++i) {
      velocity0_ += force0_*dt / mass_;
      velocity1_ += force1_*dt / mass_;
      position0_ += velocity0_*dt;
      position1_ += velocity1_*dt;
    }
  }
};

IKRA_HOST_STORAGE(Body)

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
}

void run_simulation() {
  for (int i = 0; i < kIterations; ++i) {
    // Reset forces.
    execute<kNumBodies>(&Body::reset_force);

    // Update forces.
    execute<kNumBodies>(&Body::add_force_to_all);

    // Update velocities and positions.
    execute<kNumBodies>(&Body::update, kTimeInterval);
  }
}

void run_simple() {
  for (int i = 0; i < kIterations*100; ++i) {
    execute<kNumBodies>(&Body::codengen_simple_update, kTimeInterval);
  }
}

int main() {
    // Initialize object storage.
  Body::initialize_storage();

  uint64_t time_instantiation = measure<>::execution(instantiation);
  uint64_t time_simulation = measure<>::execution(run_simulation);
  uint64_t time_simple = measure<>::execution(run_simple);

  // Calculate checksum
  int checksum = 11;
  for (uintptr_t i = 0; i < kNumBodies; i++) {
    checksum += reinterpret_cast<int>(
        r_float2int(Body::get(i)->position0_));
    checksum += reinterpret_cast<int>(
        r_float2int(Body::get(i)->position1_));
    checksum = checksum % 1234567;

    if (i < 10) {
      printf("VALUE[%lu] = %f, %f\n", i,
             (double) Body::get(i)->position0_,
             (double) Body::get(i)->position1_);
    }
  }

  printf("instantiation: %lu\nsimulation: %lu\nsimple: %lu\n checksum: %i\n",
         time_instantiation, time_simulation, time_simple, checksum);
  return 0;
}

void codegen_reset_force(Body* body) {
  body->reset_force();
}

void codegen_add_force(Body* self, Body* other) {
  self->add_force(other);
}

void codegen_update(Body* self, double dt) {
  self->update(dt);
}

void codegen_reset_force_manual(Body* body) {
  body->force0_ = 0.0;
  body->force1_ = 0.0;
}

void codegen_reset_force_half(Body* body) {
  body->force0_ = 0.0;
}

void codegen_reset_mass(Body* body) {
  body->mass_ = 0.0;
}

void codengen_simple_update(Body* body, double dt) {
  body->codengen_simple_update(dt);
}
