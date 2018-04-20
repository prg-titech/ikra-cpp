#include <limits>
#include <math.h>
#include <stdint.h>

#include "bfs_loader.h"
#include "executor/executor.h"
#include "soa/soa.h"

static const int kInlineSize = 2;
static const int kMaxVertices = 20000;
static const int kMaxEdges = 100000;

using ikra::soa::IndexType;
using ikra::soa::SoaLayout;
using ikra::soa::StaticStorageWithArena;
using ikra::soa::kAddressModeZero;
using ikra::executor::execute;
using ikra::executor::execute_and_reduce;


class Vertex : public SoaLayout<
    Vertex, kMaxVertices, kAddressModeZero,
    StaticStorageWithArena<kMaxEdges*sizeof(Vertex*)>> {
 public:
  IKRA_INITIALIZE_CLASS

  Vertex(const std::vector<IndexType>& neighbors)
      : adj_list_(neighbors.size()) {
    adj_list_size_ = neighbors.size();
    for (int i = 0; i < num_neighbors(); ++i) {
      adj_list_[i] = Vertex::get_uninitialized(neighbors[i]);
    }
  }

  int num_neighbors() {
    return adj_list_size_;
  }

  bool update_distance(int distance) {
    if (distance < distance_) {
      distance_ = distance;
      return true;
    } else {
      return false;
    }
  }

  // Visit the vertex, i.e., update the distances of all neighbors if this
  // vertex is in the frontier, as indicated by the "iteration" field. Returns
  // "true" if at least one neighbor was updated.
  bool visit(int iteration) {
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

  int_ distance_ = std::numeric_limits<int>::max();
  int_ adj_list_size_;

  // Some elements are stored in the external storage.
  array_(Vertex*, kInlineSize, partially_inlined) adj_list_;
};

IKRA_HOST_STORAGE(Vertex)


int run() {
  int counter = 0;
  bool running = true;

  for (int iteration = 0; running; ++iteration) {
    running = execute_and_reduce(&Vertex::visit,
                                 [](bool a, bool b) { return a || b; },
                                 /*default_value=*/ false,
                                 /*args...=*/ iteration);
    counter++;
  }

  return counter;
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
  Vertex::get(atoi(argv[3]))->distance_ = 0;
  // Start algorithm.
  int iterations = run();

  // Print results.
  printf("Iterations: %i\n", iterations);
  execute(&Vertex::print_distance);
}
