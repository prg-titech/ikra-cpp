#ifndef SOA_STORAGE_H
#define SOA_STORAGE_H

#include "soa/constants.h"

namespace ikra {
namespace soa {

// This class contains a pointer to the data for the owner class.
// It also keeps track of the number of created instances ("size").
template<class Owner, size_t ArenaSize>
class DynamicStorage_ {
 public:
  // Allocate data on the heap.
  DynamicStorage_() {
    // Note: ObjectSize is accessed within a function here.
    data = reinterpret_cast<char*>(
        malloc(Owner::ObjectSize::value * Owner::kCapacity));

    if (ArenaSize > 0) {
      arena_head_ = arena_base_ = reinterpret_cast<char*>(malloc(ArenaSize));
    } else {
      arena_head_ = arena_base_ = nullptr;
    }
  }

  // Use existing data allocation.
  explicit DynamicStorage_(void* ptr, void* arena_ptr = nullptr) : size(0) {
    data = reinterpret_cast<char*>(ptr);

    if (ArenaSize != 0 && arena_ptr == nullptr) {
      arena_head_ = arena_base_ = reinterpret_cast<char*>(malloc(ArenaSize));
    } else {
      arena_head_ = arena_base_ = reinterpret_cast<char*>(arena_ptr);
    }
  }

  // Allocate arena storage, i.e., storage that is not used for a SOA column.
  // For example, inlined dynamic arrays require arena storage for all elements
  // beyond InlineSize.
  void* allocate_in_arena(size_t bytes) {
    assert(arena_head_ != nullptr);
    void* result = reinterpret_cast<void*>(arena_head_);
    arena_head_ += bytes;
    return result;
  }

  // TODO: These should be private.
  IndexType size;
  char* data;

 private:
  char* arena_head_;
  char* arena_base_;
};

// This class contains a pointer to the data for the owner class.
// It also keeps track of the number of created instances ("size").
template<class Owner, size_t ArenaSize>
class StaticStorage_ {
 public:
  StaticStorage_() {
    if (ArenaSize == 0) {
      arena_head_ = nullptr;
    } else {
      arena_head_ = arena_base_;
    }
  }

  // TODO: These should be private.
  IndexType size;
  char data[Owner::ObjectSize::value * Owner::kCapacity];

  void* allocate_in_arena(size_t bytes) {
    assert(arena_head_ != nullptr);
    // Assert that arena is not full.
    assert(arena_head_ + bytes < arena_base_ + ArenaSize);
    void* result = reinterpret_cast<void*>(arena_head_);
    arena_head_ += bytes;
    return result;
  }

 private:
  char* arena_head_;
  char arena_base_[ArenaSize];
};

struct DynamicStorage {
  template<class Owner>
  using type = DynamicStorage_<Owner, 0>;
};

template<size_t ArenaSize>
struct DynamicStorageWithArena {
  template<class Owner>
  using type = DynamicStorage_<Owner, ArenaSize>;
};

struct StaticStorage {
  template<class Owner>
  using type = StaticStorage_<Owner, 0>;
};

template<size_t ArenaSize>
struct StaticStorageWithArena {
  template<class Owner>
  using type = StaticStorage_<Owner, ArenaSize>;
};

}  // namespace soa
}  // namespace ikra

#endif  // SOA_STORAGE_H
