#include <limits>
#include <math.h>
#include <stdint.h>
#include <cassert>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include "executor/cuda_executor.h"
#include "executor/executor.h"
#include "soa/soa.h"

static const int kMaxDegree = 10;
static const int kMaxVertices = 2000000;

using namespace std;
using ikra::soa::IndexType;
using ikra::soa::SoaLayout;
using ikra::executor::execute;

#include <chrono>

template<typename TimeT = std::chrono::nanoseconds>
struct measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep execution(F&& func, Args&&... args)
    {
        auto start = std::chrono::steady_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast< TimeT> 
                            (std::chrono::steady_clock::now() - start);
        return duration.count();
    }
};


struct Edge {
  int from;
  int to;
  int edge_id;

  // Dummy
  int alignment;
};

class Vertex : public SoaLayout<Vertex, kMaxVertices,
  ikra::soa::kAddressModeZero, 
  ikra::soa::StaticStorageWithArena<1024*1024*500>,
  ikra::soa::kLayoutModeSoa> {
 public:
  IKRA_INITIALIZE_CLASS

  __device__ Vertex(int num_neighbors) : adj_list_size_(num_neighbors),
      adj_list_(num_neighbors) {
    // If this check fails, we the dataset cannot be run with this
    // implementation.
    assert(num_neighbors <= kMaxDegree);
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

  // A fully inlined SOA array.
  array_(Vertex*, 10, partially_inlined) adj_list_;
};

IKRA_DEVICE_STORAGE(Vertex)


__global__ void reset_kernel() {
  unsigned int tid = blockIdx.x *blockDim.x + threadIdx.x;
  if (tid < Vertex::size()) {
    Vertex::get(tid)->distance_ = Vertex::size()+100;
  }
}

__global__ void launch_kernel_measures(int iteration) {
  unsigned int tid = blockIdx.x *blockDim.x + threadIdx.x;
  if (tid < Vertex::size()) {
    Vertex::get(tid)->visit(iteration);
  }
}

Vertex* v_first;
int num_vert;
void run_measured() {
  for (int iteration = 0; iteration < 786; ++iteration) {
    /*
    cuda_execute(&Vertex::visit,
                 v_first,
                 Vertex::size(),
                 iteration);
                 */
    //printf("%i\n",num_vert);
    launch_kernel_measures<<<num_vert/1024+1, 1024>>>(iteration);
    cudaDeviceSynchronize();
  }
}

void run(int start_vertex) {
  long long unsigned int time = 0;
  for (int r = 0; r < 100; ++r) {
    reset_kernel<<<Vertex::size()/1024+1, 1024>>>();
    gpuErrchk(cudaDeviceSynchronize());
    Vertex::get(start_vertex)->distance_ = 0;
    gpuErrchk(cudaDeviceSynchronize());

    time += measure<>::execution(run_measured);
  }

  printf("Time: %llu\n", time);
}

Edge* h_edges;
Edge* d_edges;
int* h_neighbor_sizes;
int* d_neighbor_sizes;

__global__ void create_objects(int* neighbors, IndexType num_vertices) {
  unsigned int tid = blockIdx.x *blockDim.x + threadIdx.x;

  if (tid < num_vertices) {
    Vertex* new_vertex = new(Vertex::get_uninitialized(tid))
        Vertex(neighbors[tid]);
    assert(new_vertex->id() == tid);
  }

  if (tid == 0) {
    Vertex::storage().increase_size(num_vertices);
  }
}

__global__ void create_edges(Edge* edges, IndexType num_edges) {
  unsigned int tid = blockIdx.x *blockDim.x + threadIdx.x;

  if (tid < num_edges) {
    Vertex* v_from = Vertex::get_uninitialized(edges[tid].from);
    Vertex* v_to = Vertex::get_uninitialized(edges[tid].to);
    v_from->adj_list_[edges[tid].edge_id] = v_to;
  }
}

map<int, int> m_indices;
int next_index = 0;
int get_real_index(int id) {
  if (m_indices.find(id) == m_indices.end()) {
    m_indices[id] = next_index++;
  }

  return m_indices[id];
}

void load_file(const char* filename, IndexType num_vertices, IndexType num_edges) {
  ifstream file;
  file.open(filename);
  num_vert = num_vertices;

  if (!file) {
    printf("Unable to open file: %s\n", filename);
    exit(1);
  }

  // Load from file.
  h_neighbor_sizes = new int[num_vertices]();
  h_edges = new Edge[num_edges];
  int* edge_ids = new int[num_vertices]();

  printf("Allocated host data.\n");
  fflush(stdout);

  int index = 0;
  int f_from, f_to;

  while (file >> f_from && file >> f_to) {
    int v_from = get_real_index(f_from);
    int v_to = get_real_index(f_to);

    if (v_from >= num_vertices || v_to >= num_vertices) {
      printf("Vertex out of bounds: %i or %i\n", v_from, v_to);
      exit(1);
    }

    h_edges[index].from = v_from;
    h_edges[index].to = v_to;
    h_edges[index].edge_id = edge_ids[v_from];

    h_neighbor_sizes[v_from]++;
    edge_ids[v_from]++;
    index++;
  }

  printf("Finished reading file.\n");
  fflush(stdout);

  cudaMalloc((void**) &d_neighbor_sizes, num_vertices*sizeof(int));
  cudaMemcpy(d_neighbor_sizes, h_neighbor_sizes, num_vertices*sizeof(int),
             cudaMemcpyHostToDevice);
  create_objects<<<num_vertices/1024 + 1, 1024>>>(
      d_neighbor_sizes, num_vertices);
  gpuErrchk(cudaDeviceSynchronize());

  cudaMalloc((void**) &d_edges, num_edges*sizeof(Edge));
  cudaMemcpy(d_edges, h_edges, num_edges*sizeof(Edge), cudaMemcpyHostToDevice);
  create_edges<<<num_edges/1024 + 1, 1024>>>(d_edges, num_edges);
  gpuErrchk(cudaDeviceSynchronize());
}

int main(int argc, char* argv[]) {
  // Load vertices from file.
  if (argc != 5) {
    printf("Usage: %s filename num_vertices num_edges start_vertex\n",
           argv[0]);
    exit(1);
  }

  Vertex::initialize_storage();
  load_file(argv[1], atoi(argv[2]), atoi(argv[3]));
  printf("Num vertices: %i\n", next_index);
  printf("Loading done!\n");

  // Start algorithm.
  v_first = Vertex::get_uninitialized(0);
  run(atoi(argv[4]));
  printf("Processing done!\n");

  // Note: execute is host side, cuda_execute is device side.
  for (int i = 0; i < 10; ++i) {
    printf("%i\n", (int) Vertex::get(i)->distance_);
  }
//  execute(&Vertex::print_distance);

  // Ensure nothing went wrong on the GPU.
  gpuErrchk(cudaPeekAtLastError());
}
