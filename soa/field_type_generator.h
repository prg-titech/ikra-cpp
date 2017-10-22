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
  }; \
  type ## _<OffsetCounter<field_id>::value>

#define IKRA_ARRAY_AOS_FIELD_TYPE_GENERATOR(type, size, field_id) \
  template<typename DummyT> \
  struct OffsetCounter<field_id + 1, DummyT> { \
    static const uint32_t value = OffsetCounter<field_id>::value + \
                                  sizeof(std::array<type, size>); \
    static const bool kIsSpecialization = true; \
  }; \
  array::aos<type, size, OffsetCounter<field_id>::value>

#define IKRA_ARRAY_SOA_FIELD_TYPE_GENERATOR(type, size, field_id) \
  template<typename DummyT> \
  struct OffsetCounter<field_id + 1, DummyT> { \
    static const uint32_t value = OffsetCounter<field_id>::value + \
                                  sizeof(std::array<type, size>); \
    static const bool kIsSpecialization = true; \
  }; \
  array::soa<type, size, OffsetCounter<field_id>::value>

// Generate types that keep track of offsets by themselves. Add more types
// as needed.
#define bool__(field_id) IKRA_FIELD_TYPE_GENERATOR(bool, field_id)
#define char__(field_id) IKRA_FIELD_TYPE_GENERATOR(char, field_id)
#define double__(field_id) IKRA_FIELD_TYPE_GENERATOR(double, field_id)
#define float__(field_id) IKRA_FIELD_TYPE_GENERATOR(float, field_id)
#define int__(field_id) IKRA_FIELD_TYPE_GENERATOR(int, field_id)
#define array_aos__(field_id, type, size) \
    IKRA_ARRAY_AOS_FIELD_TYPE_GENERATOR(type, size, field_id)
#define array_soa__(field_id, type, size) \
    IKRA_ARRAY_SOA_FIELD_TYPE_GENERATOR(type, size, field_id)

// Generate types that keep track of offsets and field indices by themselves.
// Add more types are needed.
#define IKRA_NEXT_FIELD_OFFSET __COUNTER__ - kCounterFirstIndex - 1
#define bool___ IKRA_FIELD_TYPE_GENERATOR(bool, IKRA_NEXT_FIELD_OFFSET)
#define char___ IKRA_FIELD_TYPE_GENERATOR(char, IKRA_NEXT_FIELD_OFFSET)
#define double___ IKRA_FIELD_TYPE_GENERATOR(double, IKRA_NEXT_FIELD_OFFSET)
#define float___ IKRA_FIELD_TYPE_GENERATOR(float, IKRA_NEXT_FIELD_OFFSET)
#define int___ IKRA_FIELD_TYPE_GENERATOR(int, IKRA_NEXT_FIELD_OFFSET)
#define array_aos___(type, size) \
    IKRA_ARRAY_AOS_FIELD_TYPE_GENERATOR(type, size, IKRA_NEXT_FIELD_OFFSET)
#define array_soa___(type, size) \
    IKRA_ARRAY_SOA_FIELD_TYPE_GENERATOR(type, size, IKRA_NEXT_FIELD_OFFSET)
