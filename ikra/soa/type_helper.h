#ifndef SOA_TYPE_HELPER_H
#define SOA_TYPE_HELPER_H

#include "soa/array_field.h"
#include "soa/constants.h"
#include "soa/partially_inlined_array_field.h"

namespace ikra {
namespace soa {

template<typename T>
struct IsArrayType {
  static const bool value = false;
};

// TODO: Do we have to enumerate all template parameters here?
template<typename T,
         typename B,
         size_t ArraySize,
         IndexType Capacity,
         uint32_t Offset,
         int AddressMode,
         int StorageMode,
         int LayoutMode,
         class Owner>
struct IsArrayType<ArrayObjectField_<
    T, B, ArraySize, Capacity, Offset, AddressMode, StorageMode,
    LayoutMode, Owner>> {
  static const bool value = true;
};

template<typename B,
         size_t ArraySize,
         IndexType Capacity,
         uint32_t Offset,
         int AddressMode,
         int StorageMode,
         int LayoutMode,
         class Owner>
struct IsArrayType<FullyInlinedArrayField_<
    B, ArraySize, Capacity, Offset, AddressMode, StorageMode,
    LayoutMode, Owner>> {
  static const bool value = true;
};

template<typename B,
         size_t InlinedSize,
         IndexType Capacity,
         uint32_t Offset,
         int AddressMode,
         int StorageMode,
         int LayoutMode,
         class Owner,
         class ArraySizeT>
struct IsArrayType<PartiallyInlinedArrayField_<
    B, InlinedSize, Capacity, Offset, AddressMode, StorageMode,
    LayoutMode, Owner, ArraySizeT>> {
  static const bool value = true;
};

}  // soa
}  // ikra

#endif  // SOA_TYPE_HELPER_H
