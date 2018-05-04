#ifndef EXECUTOR_OBJECT_RANGE_H
#define EXECUTOR_OBJECT_RANGE_H

#include "soa/constants.h"
#include "soa/cuda.h"

namespace ikra {
namespace executor {
namespace cuda {

// All objects of a given class T.
template<typename T>
class FullObjectRange {
 public:
  __ikra_device__ IndexType size() const { return T::size(); }

  __ikra_device__ T* get(IndexType index) const { return T::get(index); }
};

// A specified number of objects, starting from a specified first object.
template<typename T>
class SequenceObjectRange {
 public:
  __ikra_device__ SequenceObjectRange(T* first, IndexType num_objs)
      : first_id_(first->id()), num_objs_(num_objs) {
    assert(first_id_ + num_objs <= T::size());
  }

  __ikra_device__ IndexType size() const { return num_objs_; }

  __ikra_device__ T* get(IndexType index) const {
    return T::get(first_id_ + index);
  }

 private:
  const IndexType first_id_;
  const IndexType num_objs_;
};

// A specified number of objects, starting from a specified first object.
// The number of objects is a template parameter.
template<typename T, IndexType NumObjects>
class FixedSizeSequenceObjectRange {
 public:
  __ikra_device__ FixedSizeSequenceObjectRange(T* first)
      : first_id_(first->id()) {
    assert(first_id_ + NumObjects <= T::size());
  }

  __ikra_device__ IndexType size() const { return NumObjects; }

  __ikra_device__ T* get(IndexType index) const {
    return T::get(first_id_ + index);
  }

 private:
  const IndexType first_id_;
};

// All objects stored in a given array or array-like object.
template<typename T, typename ArrT>
class ArrayObjectRange {
 public:
  __ikra_device__ ArrayObjectRange(const ArrT& array) : array_v_(array) {}

  __ikra_device__ IndexType size() const { return array_v_.size(); };

  __ikra_device__ T* get(IndexType index) const {
    return T::get(array_v_[index]);
  }

 private:
  const ArrT& array_v_;
};

}  // cuda
}  // executor
}  // ikra

#endif  // EXECUTOR_OBJECT_RANGE_H
