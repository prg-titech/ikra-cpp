// This macro is used when generating field types. A field type in user code
// such as int__(idx) should expand to an offset counter increment and a field
// type template instantiation (e.g., int_) at the computed offset.
// Note: This macro must not be undefined even though it is only called within
// this class. It is used from other macros and cannot be called anymore when
// undefined here.
#define IKRA_FIELD_TYPE_GENERATOR(type, field_id) \
  template<typename DummyT> \
  struct OffsetCounter<field_id + 1, DummyT> { \
    static const uint32_t value = OffsetCounter<field_id>::value + \
                                  sizeof(type); \
    static const bool kIsSpecialization = true; \
    static void DBG_print_offsets() { \
      printf("OffsetCounter<%i>::value = %u\n", field_id + 1, value); \
      OffsetCounter<field_id + 2>::DBG_print_offsets(); \
    } \
  }; \
  soa_ ## type<OffsetCounter<field_id>::value>

#define IKRA_ARRAY_FIELD_TYPE_GENERATOR(type, size, layout, field_id) \
  template<typename DummyT> \
  struct OffsetCounter<field_id + 1, DummyT> { \
    static const uint32_t value = OffsetCounter<field_id>::value + \
        array::layout<type, size, OffsetCounter<field_id>::value>::kSize; \
    static const bool kIsSpecialization = true; \
    static void DBG_print_offsets() { \
      printf("OffsetCounter<%i>::value = %u    [ ARRAY ]\n", \
             field_id + 1, value); \
      OffsetCounter<field_id + 2>::DBG_print_offsets(); \
    } \
  }; \
  array::layout<type, size, OffsetCounter<field_id>::value>

#define IKRA_CUSTOM_FIELD_TYPE_GENERATOR(type, field_id) \
  template<typename DummyT> \
  struct OffsetCounter<field_id + 1, DummyT> { \
    static const uint32_t value = OffsetCounter<field_id>::value + \
                                  sizeof(type); \
    static const bool kIsSpecialization = true; \
    static void DBG_print_offsets() { \
      printf("OffsetCounter<%i>::value = %u    [ CUSTOM ]\n", \
             field_id + 1, value); \
      OffsetCounter<field_id + 2>::DBG_print_offsets(); \
    } \
  }; \
  Field<type, OffsetCounter<field_id>::value>

// Generate types that keep track of offsets by themselves. Add more types
// as needed.
#define bool__(field_id) IKRA_FIELD_TYPE_GENERATOR(bool, field_id)
#define char__(field_id) IKRA_FIELD_TYPE_GENERATOR(char, field_id)
#define double__(field_id) IKRA_FIELD_TYPE_GENERATOR(double, field_id)
#define float__(field_id) IKRA_FIELD_TYPE_GENERATOR(float, field_id)
#define int__(field_id) IKRA_FIELD_TYPE_GENERATOR(int, field_id)
#define uint8_t__(field_id) IKRA_FIELD_TYPE_GENERATOR(uint8_t, field_id)
#define uint16_t__(field_id) IKRA_FIELD_TYPE_GENERATOR(uint16_t, field_id)
#define uint32_t__(field_id) IKRA_FIELD_TYPE_GENERATOR(uint32_t, field_id)
#define uint64_t__(field_id) IKRA_FIELD_TYPE_GENERATOR(uint64_t, field_id)

#define array__(...) PP_CONCAT(ARRAY__, PP_NARG(__VA_ARGS__))(__VA_ARGS__)
#define ARRAY__4(type, size, layout, field_id) \
    IKRA_ARRAY_FIELD_TYPE_GENERATOR(type, size, layout, field_id)
#define ARRAY__3(type, size, field_id) ARRAY_3(type, size, soa, field_id)
#define ARRAY__2(arg1, arg2) \
    static_assert(false, "At least three arguments required for array__.")
#define ARRAY__1(arg1) ARRAY_2(0, 0)
#define ARRAY__0() ARRAY_2(0, 0)

#define field__(type, field_id) \
    IKRA_CUSTOM_FIELD_TYPE_GENERATOR(type, field_id)
#define ref__(type, field_id) \
    IKRA_CUSTOM_FIELD_TYPE_GENERATOR(type*, field_id)

// Generate types that keep track of offsets and field indices by themselves.
// Add more types are needed.
#define IKRA_NEXT_FIELD_ID __COUNTER__ - kCounterFirstIndex - 1
#define bool_ IKRA_FIELD_TYPE_GENERATOR(bool, IKRA_NEXT_FIELD_ID)
#define char_ IKRA_FIELD_TYPE_GENERATOR(char, IKRA_NEXT_FIELD_ID)
#define double_ IKRA_FIELD_TYPE_GENERATOR(double, IKRA_NEXT_FIELD_ID)
#define float_ IKRA_FIELD_TYPE_GENERATOR(float, IKRA_NEXT_FIELD_ID)
#define int_ IKRA_FIELD_TYPE_GENERATOR(int, IKRA_NEXT_FIELD_ID)
#define uint8_t_ IKRA_FIELD_TYPE_GENERATOR(uint8_t, IKRA_NEXT_FIELD_ID)
#define uint16_t_ IKRA_FIELD_TYPE_GENERATOR(uint16_t, IKRA_NEXT_FIELD_ID)
#define uint32_t_ IKRA_FIELD_TYPE_GENERATOR(uint32_t, IKRA_NEXT_FIELD_ID)
#define uint64_t_ IKRA_FIELD_TYPE_GENERATOR(uint64_t, IKRA_NEXT_FIELD_ID)

#define array_(...) PP_CONCAT(ARRAY_, PP_NARG(__VA_ARGS__))(__VA_ARGS__)
#define ARRAY_3(type, size, layout) \
    IKRA_ARRAY_FIELD_TYPE_GENERATOR(type, size, layout, IKRA_NEXT_FIELD_ID)
#define ARRAY_2(type, size) ARRAY_3(type, size, soa)
#define ARRAY_1(arg) \
    static_assert(false, "At least two arguments required for array_.")
#define ARRAY_0() ARRAY_1(0)

#define field_(type) IKRA_CUSTOM_FIELD_TYPE_GENERATOR(type, IKRA_NEXT_FIELD_ID)
#define ref_(type) IKRA_CUSTOM_FIELD_TYPE_GENERATOR(type*, IKRA_NEXT_FIELD_ID)
