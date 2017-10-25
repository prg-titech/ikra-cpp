#ifndef SOA_STORAGE_H
#define SOA_STORAGE_H

#include "soa/constants.h"

namespace ikra {
namespace soa {

// This class contains a pointer to the data for the owner class.
// It also keeps track of the number of created instances ("size").
template<class Owner>
class DynamicStorage {
 public:
  // Allocate data on the heap.
  DynamicStorage() {
    // Note: ObjectSize is accessed within a function here.
    data = reinterpret_cast<char*>(
        malloc(Owner::ObjectSize::value * Owner::kCapacity));
  }

  // Use existing data allocation.
  explicit DynamicStorage(void* ptr) : size(0) {
    data = reinterpret_cast<char*>(ptr);
  }

  IndexType size;
  char* data;
};

// This class contains a pointer to the data for the owner class.
// It also keeps track of the number of created instances ("size").
template<class Owner>
class StaticStorage {
 public:
  IndexType size;
  char data[Owner::ObjectSize::value * Owner::kCapacity];
};

}  // namespace soa
}  // namespace ikra

#endif  // SOA_STORAGE_H
