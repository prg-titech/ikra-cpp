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

// This helper macro can be used to allow reinterpret_casts in constexpr.
// Only needed for clang.
#define IKRA_fold(x) (__builtin_constant_p(x) ? (x) : (x))

// This applies the macro to all the arithmetic and bitwise operators.
#define IKRA_DEFINE_ALL_OPERATORS(macro) \
macro(+); macro(-); macro(*); macro(/); macro(%); \
macro(&); macro(|); macro(^); macro(<<); macro(>>);

// This class is used to check if the compiler supports the selected addressing
// mode.
template<uintptr_t N>
class AddressingModeCompilerCheck {
  char ___[N];
};

}  // soa
}  // ikra

#endif  // SOA_UTIL_H
