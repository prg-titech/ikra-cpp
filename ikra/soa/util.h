#ifndef SOA_UTIL_H
#define SOA_UTIL_H

namespace ikra {
namespace soa {

// Helper structure for checking if a storage buffer has a minimum required
// size. The structure is designed in such a way that the compiler may display
// the required size as part of the error message.
template<size_t StorageBufferSize, size_t RequiredMinimumSize>
struct StorageSizeCheck {
  static_assert(StorageBufferSize >= RequiredMinimumSize,
      "Storage buffer size too small. Must be at least sizeof(Storage)");
  static constexpr bool value = StorageBufferSize >= RequiredMinimumSize;
};

}  // soa
}  // ikra

#endif  // SOA_UTIL_H
