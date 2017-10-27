// Particle-Well Simulation
// Code adapted from: http://physics.princeton.edu/~fpretori/Nbody/code.htm

#include <math.h>

#include "SDL2/SDL.h"

#include "executor/executor.h"
#include "soa/soa.h"

using ikra::executor::execute;
using ikra::executor::Iterator;
using ikra::soa::SoaLayout;

static const int kIterations = 10000;
static const int kNumWells = 1;
static const int kNumParticles = 20;
static const double kTimeInterval = 1500;

static const double kGravityConstant = 6.673e-11;   // gravitational constant
static const double kSolarMass = 1.98892e30;
static const double kEarthMass = 5.972e24;
static const double kEarthSunDistance = 149600000000;
static const double kEarthVelocity = 30000;

static const int kWindowWidth = 1000;
static const int kWindowlHeight = 1000;
static const double kScalingFactor = 2.5e-9;

// Draw a rectangle with equal height/width at a given position.
static void render_rect(SDL_Renderer* renderer, double x, double y, int side) {
  SDL_Rect rect;
  rect.x = x*kScalingFactor + kWindowWidth/2 - side/2;
  rect.y = y*kScalingFactor + kWindowlHeight/2 - side/2;
  rect.w = side;
  rect.h = side;
  SDL_RenderDrawRect(renderer, &rect);
}

class Particle;

class Well : public SoaLayout<Well, kNumWells> {
 public:
  #include IKRA_INITIALIZE_CLASS

  Well(double mass, double pos_x, double pos_y) : mass_(mass) {
    position_[0] = pos_x;
    position_[1] = pos_y;
  }

  void add_force();

  void draw(SDL_Renderer* renderer) {
    render_rect(renderer, position_[0], position_[1], 50);
    render_rect(renderer, position_[0], position_[1], 60);
  }

  double_ mass_;
  array_(double, 2) position_;
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

  double_ mass_;
  array_(double, 2) position_;
  array_(double, 2) velocity_;
  array_(double, 2) force_;

  double distance_to(Well* well) {
    double dx = position_[0] - well->position_[0];
    double dy = position_[1] - well->position_[1];
    return sqrt(dx*dx + dy*dy);
  }

  void add_force(Well* well) {
    double EPS = 3e4;    // Softening parameter (just to avoid infinities).
    double dx = well->position_[0] - position_[0];
    double dy = well->position_[1] - position_[1];
    double dist = sqrt(dx*dx + dy*dy);
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

  void draw(SDL_Renderer* renderer) {
    render_rect(renderer, position_[0], position_[1], 30);
  }
};

Particle::Storage Particle::storage;


// Defined here because definition depends on Particle.
void Well::add_force() {
  // Add force induced by this well to all particles.
  execute(&Particle::add_force, this);
}


int main() {
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    printf("Could not initialize SDL!\n");
    exit(1);
  }

  SDL_Window* window = nullptr;
  SDL_Renderer* renderer = nullptr;

  if (SDL_CreateWindowAndRenderer(kWindowWidth, kWindowWidth, 0,
        &window, &renderer) != 0) {
    printf("Could not create window/render!\n");
    exit(1);
  }

  // Create objects.
  for (int i = 0; i < kNumWells; ++i) {
    new Well(kSolarMass, 0, 0);
  }

  for (int i = 0; i < kNumParticles; ++i) {
    new Particle(kEarthMass, kEarthSunDistance, 0, 0, kEarthVelocity);
  }

  SDL_bool done = SDL_FALSE;
  while (!done) {
    // Reset forces.
    execute(&Particle::reset_force);

    // Update forces.
    execute(&Well::add_force);

    // Update velocities and positions.
    execute(&Particle::update, kTimeInterval);

    // Draw scene.
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    execute(&Well::draw, renderer);
    execute(&Particle::draw, renderer);
    SDL_RenderPresent(renderer);

    // Wait for user to close the window.
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        done = SDL_TRUE;
      }
    }
  }

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}
