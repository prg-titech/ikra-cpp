#ifndef SOA_SOA_H
#define SOA_SOA_H

#if __cplusplus < 201103L
  #error This library needs at least a C++11 compliant compiler.
#endif

#if defined(__clang__)
  #warning Loop vectorization is broken when using clang.
#elif defined(__GNUC__) || defined(__GNUG__)
  // Everything OK with gcc.
#else
  #warning ikra-cpp was not tested with this compiler.
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// Asserts active only in debug mode (NDEBUG).
#include <cassert>

#include "executor/iterator.h"
#include "soa/array_field.h"
#include "soa/array_operator.h"
#include "soa/constants.h"
#include "soa/field.h"
#include "soa/field_type_generator.h"
#include "soa/inlined_dynamic_array_field.h"
#include "soa/layout.h"
#include "soa/preprocessor.h"
#include "soa/storage.h"

#endif  // SOA_SOA_H
