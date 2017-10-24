// Particle-Well Simulation
// Code adapted from: http://physics.princeton.edu/~fpretori/Nbody/code.htm

#include <math.h>
#include "executor/executor.h"
#include "executor/iterator.h"
#include "soa/soa.h"

using ikra::executor::execute;
using ikra::executor::Iterator;
using ikra::soa::SoaLayout;

static const int kIterations = 10000;
static const int kNumWells = 1;
static const int kNumParticles = 20;
static const double kTimeInterval = 100;

static const double kGravityConstant = 6.673e-11;   // gravitational constant
static const double kSolarMass = 1.98892e30;
static const double kEarthMass = 5.972e24;
static const double kEarthSunDistance = 149600000000;
static const double kEarthVelocity = 30000;

class Particle;

class Well : public SoaLayout<Well, kNumWells> {
 public:
  #include IKRA_INITIALIZE_CLASS

  Well(double mass, double pos_x, double pos_y) : mass_(mass) {
    position_[0] = pos_x;
    position_[1] = pos_y;
  }

  void add_force();

  double___ mass_;
  array_soa___(double, 2) position_;
};

Well::Storage Well::storage;


class Particle : public SoaLayout<Particle, kNumParticles> {
 public:
  #include IKRA_INITIALIZE_CLASS

  Particle(double mass, double pos_x, double pos_y, double vel_x, double vel_y)
      : mass_(mass) {
    position_[0] = pos_x;
    position_[1] = pos_y;
    velocity_[0] = vel_x;
    velocity_[1] = vel_y;
    this->reset_force();
  }

  double___ mass_;
  array_soa___(double, 2) position_;
  array_soa___(double, 2) velocity_;
  array_soa___(double, 2) force_;

  double distance_to(Well* well) {
    double dx = position_[0] - well->position_[0];
    double dy = position_[1] - well->position_[1];
    return sqrt(dx*dx + dy*dy);
  }

  void add_force(Well* well) {
    double EPS = 3e4;    // Softening parameter (just to avoid infinities).
    double dx = well->position_[0] - position_[0];
    double dy = well->position_[1] - position_[1];
    double dist = sqrt(dx*dx + dy+dy);
    double F = kGravityConstant*mass_*well->mass_ / (dist*dist + EPS*EPS);
    force_[0] += F*dx / dist;
    force_[1] += F*dy / dist;
  }

  void reset_force() {
    force_[0] = 0.0f;
    force_[1] = 0.0f;
  }

  void update(double dt) {
    velocity_[0] += force_[0]*dt / mass_;
    velocity_[1] += force_[1]*dt / mass_;
    position_[0] += velocity_[0]*dt;
    position_[1] += velocity_[1]*dt;
  }
};

Particle::Storage Particle::storage;


void Well::add_force() {
  // Add force induced by this well to all particles.
  execute<Particle>(&Particle::add_force, this);
}


int main() {
  // Create objects.
  for (int i = 0; i < kNumWells; ++i) {
    new Well(kSolarMass, 0, 0);
  }

  for (int i = 0; i < kNumParticles; ++i) {
    new Particle(kEarthMass, kEarthSunDistance, 0, 0, kEarthVelocity);
  }

  for (int i = 0; i < kIterations; ++i) {
    // Reset forces.
    execute<Particle>(&Particle::reset_force);

    // Update forces.
    execute<Well>(&Well::add_force);

    // Update velocities and positions.
    execute<Particle>(&Particle::update, kTimeInterval);

    // TODO: Draw to screen.
    printf("Particle 1 pos: %f, %f\n",
           (float) Particle::get(0)->position_[0],
           (float) Particle::get(0)->position_[1]);
  }
}
