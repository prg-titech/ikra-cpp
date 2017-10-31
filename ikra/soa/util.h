#ifndef SOA_UTIL_H
#define SOA_UTIL_H

namespace ikra {
namespace soa {

// Returns the size (number of elements) of a char array.
template<size_t n>
static char (&char_array_size(const char (&)[n]))[n];

}  // soa
}  // ikra

#endif  // SOA_UTIL_H
