# Ikra-Cpp
Ikra-Cpp is a C++/CUDA DSL for object-oriented programming with Structure of Arrays (SOA) data layout. Ikra-Cpp provides an AOS-style OOP notation that is similar to standard C++ but layouts data as SOA. This gives programmers the performance benefit of SOA and the expressivness of AOS-style object-oriented programming at the same time.

## Example
```c++
// N-Body Simulation
// Code adapted from: http://physics.princeton.edu/~fpretori/Nbody/code.htm

#include "executor/executor.h"
#include "soa/soa.h"

using namespace std;
using ikra::executor::execute;
using ikra::soa::SoaLayout;

static const int kNumBodies = 25;
static const double kMaxMass = 1000;
static const double kTimeInterval = 0.5;
static const double kGravityConstant = 6.673e-11;   // gravitational constant
static const int kIterations = 1000;

#define RAND (1.0 * rand() / RAND_MAX)

class Body : public SoaLayout<Body, kNumBodies> {
 public:
  IKRA_INITIALIZE_CLASS

  Body(double mass, double pos_x, double pos_y, double vel_x, double vel_y)
      : mass_(mass) {
    position_[0] = pos_x;
    position_[1] = pos_y;
    velocity_[0] = vel_x;
    velocity_[1] = vel_y;
    this->reset_force();
  }

  double_ mass_;
  array_(double, 2) position_;
  array_(double, 2) velocity_;
  array_(double, 2) force_;

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
    execute(&Body::add_force, this);
  }

  void reset_force() {
    force_[0] = 0;
    force_[1] = 0;
  }

  void update(double dt) {
    velocity_[0] += force_[0]*dt / mass_;
    velocity_[1] += force_[1]*dt / mass_;
    position_[0] += velocity_[0]*dt;
    position_[1] += velocity_[1]*dt;

    // Reflect bodies.
    if (position_[0] < -1 || position_[0] > 1) velocity_[0] = -velocity_[0];
    if (position_[1] < -1 || position_[1] > 1) velocity_[1] = -velocity_[1];
  }
};
IKRA_HOST_STORAGE(Body)

int main() {
  // Initialize object storage.
  Body::initialize_storage();

  // Create objects.
  for (int i = 0; i < kNumBodies; ++i) {
    double mass = (RAND/2 + 0.5) * kMaxMass;
    double pos_x = RAND*2 - 1;
    double pos_y = RAND*2 - 1;
    double vel_x = (RAND - 0.5) / 1000;
    double vel_y = (RAND - 0.5) / 1000;
    new Body(mass, pos_x, pos_y, vel_x, vel_y);
  }

  // Run simulation.
  for (int i = 0; i < kIterations; ++i) {
    // Reset forces.
    execute(&Body::reset_force);

    // Update forces.
    execute(&Body::add_force_to_all);

    // Update velocities and positions.
    execute(&Body::update, kTimeInterval);
  }
  
  return 0;
}
```

## Installation
### Prerequisites
* gcc version 5.4.0 or higher
* For visual examples (n-body simulation): SDL 2.0
* For GPU support: CUDA Toolkit 9.0 or higher
* C++11 compiler support in CPU mode, C++14 compiler support in GPU mode

### Installation via Git
``` sh
git clone --recursive git@github.com:prg-titech/ikra-cpp.git
cmake CMakeLists.txt
make
# Check if everything is working.
ctest
bin/n_body
```

