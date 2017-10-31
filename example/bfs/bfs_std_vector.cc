#include <limits>
#include <math.h>
#include <stdint.h>
#include <vector>

#include "bfs_loader.h"
#include "executor/executor.h"
#include "soa/soa.h"

static const int kMaxVertices = 20000;
char storage_buffer[50000];

using ikra::soa::IndexType;
using ikra::soa::SoaLayout;
using ikra::executor::execute;
using ikra::executor::execute_and_reduce;

class Vertex : public SoaLayout<Vertex, kMaxVertices> {
 public:
  IKRA_INITIALIZE_CLASS(storage_buffer)

  Vertex(const std::vector<IndexType>& neighbors) {
    adj_list_size_ = neighbors.size();

    for (int i = 0; i < num_neighbors(); ++i) {
      adj_list_->push_back(Vertex::get_uninitialized(neighbors[i]));
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
      for (int i = 0; i < adj_list_size_; ++i) {
        Vertex* neighbor = (*adj_list_)[i]; //->operator[](i);
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
  field_(std::vector<Vertex*>) adj_list_;
};


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
  // Ensure that the std::vector is layouted correctly.
  static_assert(
      Vertex::ObjectSize::value == sizeof(std::vector<void*>) + sizeof(int)*2,
      "Expected different SOA object size.");

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
