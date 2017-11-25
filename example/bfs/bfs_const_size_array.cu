#include <limits>
#include <math.h>
#include <stdint.h>

#include "bfs_loader.h"
#include "executor/cuda_executor.h"
#include "executor/executor.h"
#include "soa/soa.h"

static const int kMaxDegree = 10;
static const int kMaxVertices = 20000;

using ikra::soa::IndexType;
using ikra::soa::SoaLayout;
using ikra::executor::execute;

class Vertex : public SoaLayout<Vertex, kMaxVertices> {
 public:
  IKRA_INITIALIZE_CLASS

  Vertex(const std::vector<IndexType>& neighbors) {
    // If this check fails, we the dataset cannot be run with this
    // implementation.
    assert(neighbors.size() <= kMaxDegree);
    adj_list_size_ = neighbors.size();

    for (int i = 0; i < num_neighbors(); ++i) {
      Vertex* vertex = Vertex::get_uninitialized(neighbors[i]);
      adj_list_[i] = vertex;
    }
  }

  __host__ __device__ int num_neighbors() {
    return adj_list_size_;
  }

  // Visit the vertex, i.e., update the distances of all neighbors if this
  // vertex is in the frontier, as indicated by the "iteration" field. Returns
  // "true" if at least one neighbor was updated.
  __device__ bool visit(int iteration) {
    bool updated = false;

    if (distance_ == iteration) {
      for (int i = 0; i < num_neighbors(); ++i) {
        Vertex* neighbor = adj_list_[i];
        updated |= neighbor->update_distance(distance_ + 1);
      }
    }

    return updated;
  }

  void print_distance() {
    printf("distance[%lu] = %i\n", id(), (int) distance_);
  }

  void set_distance(int value) {
    distance_ = value;
  }

  __device__ bool update_distance(int distance) {
    if (distance < distance_) {
      distance_ = distance;
      return true;
    } else {
      return false;
    }
  }

  int_ distance_ = std::numeric_limits<int>::max();
  int_ adj_list_size_;

  // By default a SOA array.
  array_(Vertex*, kMaxDegree) adj_list_;
};

IKRA_DEVICE_STORAGE(Vertex)


int run() {
  int iteration = 0;
  bool running = true;

  while (running) {
    auto reducer = [](bool a, bool b) { return a || b; };
    running = cuda_execute_and_reduce(&Vertex::visit,
                                      reducer,
                                      iteration);

    ++iteration;
  }

  return iteration;
}

int main(int argc, char* argv[]) {
  // Load vertices from file.
  if (argc != 4) {
    printf("Usage: %s filename num_vertices start_vertex\n", argv[0]);
    exit(1);
  }

  Vertex::initialize_storage();
  load_file<Vertex>(argv[1], atoi(argv[2]));

  // Set start vertex.
  Vertex* start_vertex = Vertex::get(atoi(argv[3]));
  start_vertex->set_distance(0);

  // Start algorithm.
  int iterations = run();

  // Note: execute is host side, cuda_execute is device side.
  printf("Iterations: %i\n", iterations);
  execute(&Vertex::print_distance);

  // Ensure nothing went wrong on the GPU.
  gpuErrchk(cudaPeekAtLastError());
}
