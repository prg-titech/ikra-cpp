#ifndef EXAMPLE_BFS_BFS_LOADER_H
#define EXAMPLE_BFS_BFS_LOADER_H

#include <cassert>
#include <fstream>
#include <string>
#include <vector>

#include "soa/soa.h"

using namespace std;
using ikra::soa::IndexType;

template<typename VertexT>
void load_file(const char* filename, IndexType num_vertices) {
  ifstream file;
  file.open(filename);

  if (!file) {
    printf("Unable to open file: %s\n", filename);
    exit(1);
  }

  // Build adj. list until we see a new vertex.
  vector<IndexType> adj_list;
  IndexType last_v_from = 0;
  IndexType v_from, v_to;

  while (file >> v_from && file >> v_to) {
    // We assume that the edges are soreted by edge source.
    assert(v_from >= last_v_from);

    for (; last_v_from < v_from; ++last_v_from) {
      VertexT* vertex = new VertexT(adj_list);
      assert(vertex->id() == last_v_from);

      adj_list.clear();
    }

    adj_list.push_back(v_to);
  }

  for (; last_v_from < num_vertices; ++last_v_from) {
    VertexT* vertex = new VertexT(adj_list);
    assert(vertex->id() == last_v_from);

    adj_list.clear();
  }
}

#endif  // EXAMPLE_BFS_BFS_LOADER_H
