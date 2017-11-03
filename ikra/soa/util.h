#ifndef SOA_UTIL_H
#define SOA_UTIL_H


namespace ikra {
namespace soa {

// This class exports its single template parameter as "type". Used as
// workaround for an nvcc bug with friend declarations.
template<typename T>
struct NvccWorkaroundIdentityClassHolder {
  using type = T;
};

}  // soa
}  // ikra

#endif  // SOA_UTIL_H
