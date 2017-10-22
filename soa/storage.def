// This class contains a pointer to the data for the enclosing class "Self".
// It also keeps track of the number of created instances ("size").
class InternalStorage {
 public:
  // Allocate data on the heap.
  InternalStorage() {
    data = reinterpret_cast<char*>(
        malloc(Self::ObjectSize::value * ContainerSize));
  }

  // Use existing data allocation.
  explicit InternalStorage(void* ptr) : size(0) {
    data = reinterpret_cast<char*>(ptr);
  }

  // Return an iterator pointing to the first element.
  executor::Iterator_<Self*> begin() {
    return executor::Iterator<Self*>(Self::get(0));
  }

  // Return an iterator pointing to the last element.
  executor::Iterator_<Self*> end() {
    return executor::Iterator<Self*>(Self::get(size));
  }

  uint32_t size;
  char* data;
};

using Storage = InternalStorage;
