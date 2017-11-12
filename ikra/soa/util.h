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

#define IKRA_fold(x) (__builtin_constant_p(x) ? (x) : (x))

}  // soa
}  // ikra

#endif  // SOA_UTIL_H
