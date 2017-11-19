// N-Body Simulation
// Code adapted from: http://physics.princeton.edu/~fpretori/Nbody/code.htm

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <array>

#include "benchmark.h"

using namespace std;

static const int kIterations = 50000000;
static const double kMaxMass = 1000;
static const double kTimeInterval = 0.5;

static const double kGravityConstant = 6.673e-11;   // gravitational constant

#define RAND (1.0 * rand() / RAND_MAX)

char Body_arena[kNumBodies*8*20];
char* Body_arena_head = Body_arena;

class Body {
 public:
  Body(double mass, double pos_x, double pos_y, double vel_x, double vel_y) {
    position_[0] = pos_x;
    position_[1] = pos_y;
    velocity_[0] = vel_x;
    velocity_[1] = vel_y;
  }

  std::array<double, 2> position_;
  std::array<double, 2> velocity_;

  // Uncomment for AOS-32
  // (=32 additional floats, equal to 16 additional doubles)
  // std::array<double, 16> force_;


  void codengen_simple_update(double dt) {
    position_[0] += velocity_[0]*dt;
    position_[1] += velocity_[1]*dt;
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

void run_simple() {
  int iterations = kIterations * getR();
  for (int i = 0; i < iterations; ++i) {
    for (int j = 0; j < kNumBodies; ++j) {
      Body* body = reinterpret_cast<Body*>(Body_arena + sizeof(Body)*j);
      body->codengen_simple_update(kTimeInterval);
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
