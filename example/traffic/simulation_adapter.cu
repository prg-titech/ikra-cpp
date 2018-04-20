#include "simulation_adapter.h"

namespace simulation {
namespace standard {
class Simulation;
Simulation* instance;
}  // namespace standard

namespace aos_int {
class Simulation;
Simulation* instance;
}  // namespace aos_int

namespace aos_int_cuda {
void initialize();
void step();
}  // namespace aos_int_cuda
}  // namespace simulation

#include "traffic-simulation/option_standard.inc"
#include "traffic-simulation/graphml_simulation.inc"

using namespace std;

#include "traffic-simulation/traffic_aos_int.h"

void load_simulation(int argc, char** argv, unsigned int num_cars) {
  instance = build_simulation(argc, argv, num_cars);
  simulation::aos_int::instance = new simulation::aos_int::Simulation(
      simulation::standard::instance);
  simulation::aos_int_cuda::initialize();
}

