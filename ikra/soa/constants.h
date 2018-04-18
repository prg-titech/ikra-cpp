#ifndef SOA_CONSTANTS_H
#define SOA_CONSTANTS_H

#include <cstdint>

namespace ikra {
namespace soa {

// In Zero Addressing Mode, the address of an object is its ID. E.g., the
// address of the first object (ID 0) is nullptr. Pointer arithmetics is not
// possible in Zero Addressing Mode, because the size of a SOA class is zero.
// I.e., sizeof(MyClass) = 0. Increasing the size to 1 does not work, because
// instance create will attempt to zero-initialize data at an invalid address.
// Zero Addressing Mode results in efficient assembly code for field reads/
// writes. The address of a field `i` in object with ID t is defined as:
// this*sizeof(type(i)) + Capacity*Offset(i) + ClassStorageBase.
// Notice that the second operand is a compile-time constant if the container
// size (max. number of elements of a class) is a compile-time constant.
// See test/soa/benchmarks/codegen_test.cc for inspection of assembly code.
static const int kAddressModeZero = 0;

// In Valid Addressing Mode, the address of an object is the address of its
// first field, i.e., the beginning of the class storage data chunk
// (ClassStorageBase) plus ID*sizeof(first field type). The size of an object
// now defined as the size of the first field, allowing for pointer arithmetics
// on objects. Generated assembly code is less efficient, because the address
// of a field is now complex:
// (this - ClassStorageBase - sizeof(first field type))
//         / sizeof(first field type) * sizeof(type(i)) + 
//     Capacity*Offset(i) + ClassStorageBase
// See test/soa/pointer_arithmetics_test.cc for pointer arithmetics examples.
static const int kAddressModeValid = 1;

static const int kLayoutModeAos = 0;

static const int kLayoutModeSoa = 1;

// The type that is used to represent indices.
using IndexType = uintptr_t;

}  // namespace soa
}  // namespace ikra

#endif  // SOA_CONSTANTS_H
