#ifndef SOA_SOA_H
#define SOA_SOA_H

#include <tuple>
#include <type_traits>

// Asserts active only in debug mode (NDEBUG).
#include <cassert>

#include "executor/iterator.h"
#include "soa/constants.h"
#include "soa/array_field.h"
#include "soa/field.h"
#include "soa/layout.h"

#define IKRA_INITIALIZE_CLASS "soa/class_initialization.def"

#endif  // SOA_SOA_H
