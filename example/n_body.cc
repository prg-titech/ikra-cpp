// N-Body Simulation
// Code adapted from: http://physics.princeton.edu/~fpretori/Nbody/code.htm

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "SDL2/SDL.h"

#include "executor/executor.h"
#include "soa/soa.h"

using namespace std;
using ikra::executor::execute;
using ikra::executor::Iterator;
using ikra::soa::SoaLayout;

static const int kNumBodies = 25;
static const double kMaxMass = 1000;
static const double kTimeInterval = 0.5;

static const double kGravityConstant = 6.673e-11;   // gravitational constant

static const int kWindowWidth = 1000;
static const int kWindowlHeight = 1000;
static const int kMaxRect = 20;

#define RAND (1.0 * rand() / RAND_MAX)

static void render_rect(SDL_Renderer* renderer, double x, double y, double mass) {
  SDL_Rect rect;
  rect.w = rect.h = mass / kMaxMass * kMaxRect;
  rect.x = (x/2 + 0.5) * kWindowWidth - rect.w/2;
  rect.y = (y/2 + 0.5) * kWindowlHeight - rect.h/2;
  SDL_RenderDrawRect(renderer, &rect);
}

class Body : public SoaLayout<Body, kNumBodies> {
 public:
  #include IKRA_INITIALIZE_CLASS

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

    if (position_[0] < -1 || position_[0] > 1) {
      velocity_[0] = -velocity_[0];
    }
    if (position_[1] < -1 || position_[1] > 1) {
      velocity_[1] = -velocity_[1];
    }
  }

  void draw(SDL_Renderer* renderer) {
    render_rect(renderer, position_[0], position_[1], mass_);
  }
};

Body::Storage Body::storage;


int main() {
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    printf("Could not initialize SDL!\n");
    exit(1);
  }

  SDL_Window* window = nullptr;
  SDL_Renderer* renderer = nullptr;

  if (SDL_CreateWindowAndRenderer(kWindowWidth, kWindowlHeight, 0,
        &window, &renderer) != 0) {
    printf("Could not create window/render!\n");
    exit(1);
  }

  // Create objects.
  for (int i = 0; i < kNumBodies; ++i) {
    double mass = (RAND/2 + 0.5) * kMaxMass;
    double pos_x = RAND*2 - 1;
    double pos_y = RAND*2 - 1;
    double vel_x = (RAND - 0.5) / 1000;
    double vel_y = (RAND - 0.5) / 1000;
    new Body(mass, pos_x, pos_y, vel_x, vel_y);
  }

  SDL_bool done = SDL_FALSE;
  while (!done) {
    // Reset forces.
    execute(&Body::reset_force);

    // Update forces.
    execute(&Body::add_force_to_all);

    // Update velocities and positions.
    execute(&Body::update, kTimeInterval);

    // Draw scene.
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    execute(&Body::draw, renderer);
    SDL_RenderPresent(renderer);
    // done = SDL_TRUE;

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
