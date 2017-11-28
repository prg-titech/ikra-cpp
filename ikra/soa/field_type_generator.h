// This macro is used when generating field types. A field type in user code
// such as int__(idx) should expand to an offset counter increment and a field
// type template instantiation (e.g., int_) at the computed offset.
// Note: This macro must not be undefined even though it is only called within
// this class. It is used from other macros and cannot be called anymore when
// undefined here.
#if defined(__CUDA_ARCH__) || !defined(__CUDACC__)
#define IKRA_FIELD_TYPE_GENERATOR(name, type, field_id) \
  template<typename DummyT> \
  struct OffsetCounter<field_id + 1, DummyT> { \
    static const uint32_t value = OffsetCounter<field_id>::value + \
                                  sizeof(type); \
    static const bool kIsSpecialization = true; \
  }; \
  soa_ ## type<OffsetCounter<field_id>::value> name; \
  __ikra_device__ type& get_ ## name() { \
    return *name.data_ptr(this); \
  } \
  __ikra_device__ void set_ ## name(type value) { \
    *name.data_ptr(this) = value; \
  }
#else
#define IKRA_FIELD_TYPE_GENERATOR(name, type, field_id) \
  template<typename DummyT> \
  struct OffsetCounter<field_id + 1, DummyT> { \
    static const uint32_t value = OffsetCounter<field_id>::value + \
                                  sizeof(type); \
    static const bool kIsSpecialization = true; \
  }; \
  soa_ ## type<OffsetCounter<field_id>::value> name; \
  __ikra_device__ type& get_ ## name() { \
    return name.get(); \
  } \
  __ikra_device__ void set_ ## name(type value) { \
    name = value; \
  }
#endif

#define IKRA_ARRAY_FIELD_TYPE_GENERATOR(type, size, layout, field_id) \
  template<typename DummyT> \
  struct OffsetCounter<field_id + 1, DummyT> { \
    static const uint32_t value = OffsetCounter<field_id>::value + \
        array::layout<type, size, OffsetCounter<field_id>::value>::kSize; \
    static const bool kIsSpecialization = true; \
  }; \
  array::layout<type, size, OffsetCounter<field_id>::value>

#define IKRA_CUSTOM_FIELD_TYPE_GENERATOR(type, field_id) \
  template<typename DummyT> \
  struct OffsetCounter<field_id + 1, DummyT> { \
    static const uint32_t value = OffsetCounter<field_id>::value + \
                                  sizeof(type); \
    static const bool kIsSpecialization = true; \
  }; \
  Field<type, OffsetCounter<field_id>::value>

// Generate types that keep track of offsets by themselves. Add more types
// as needed.
#define bool__(name, field_id) IKRA_FIELD_TYPE_GENERATOR(name, bool, field_id)
#define char__(name, field_id) IKRA_FIELD_TYPE_GENERATOR(name, char, field_id)
#define double__(name, field_id) \
    IKRA_FIELD_TYPE_GENERATOR(name, double, field_id)
#define float__(name, field_id) \
    IKRA_FIELD_TYPE_GENERATOR(name, float, field_id)
#define int__(name, field_id) IKRA_FIELD_TYPE_GENERATOR(name, int, field_id)

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
#define bool_(name) IKRA_FIELD_TYPE_GENERATOR(name, bool, IKRA_NEXT_FIELD_ID)
#define char_(name) IKRA_FIELD_TYPE_GENERATOR(name, char, IKRA_NEXT_FIELD_ID)
#define double_(name) \
    IKRA_FIELD_TYPE_GENERATOR(name, double, IKRA_NEXT_FIELD_ID)
#define float_(name) IKRA_FIELD_TYPE_GENERATOR(name, float, IKRA_NEXT_FIELD_ID)
#define int_(name) IKRA_FIELD_TYPE_GENERATOR(name, int, IKRA_NEXT_FIELD_ID)

#define array_(...) PP_CONCAT(ARRAY_, PP_NARG(__VA_ARGS__))(__VA_ARGS__)
#define ARRAY_3(type, size, layout) \
    IKRA_ARRAY_FIELD_TYPE_GENERATOR(type, size, layout, IKRA_NEXT_FIELD_ID)
#define ARRAY_2(type, size) ARRAY_3(type, size, soa)
#define ARRAY_1(arg) \
    static_assert(false, "At least two arguments required for array_.")
#define ARRAY_0() ARRAY_1(0)

#define field_(type) IKRA_CUSTOM_FIELD_TYPE_GENERATOR(type, IKRA_NEXT_FIELD_ID)
#define ref_(type) IKRA_CUSTOM_FIELD_TYPE_GENERATOR(type*, IKRA_NEXT_FIELD_ID)
