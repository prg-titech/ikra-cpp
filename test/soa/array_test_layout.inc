class IKRA_TEST_CLASSNAME
    : public SoaLayout<IKRA_TEST_CLASSNAME, kClassMaxInst,
                       IKRA_TEST_ADDRESS_MODE> {
 public:
  #include IKRA_INITIALIZE_CLASS

  int___ field0;

  // Array has size 12 bytes.
  array_aos___(int, 3) field1;
  int___ field2;

  // Array has size 12 bytes.
  array_soa___(int, 3) field3;
  int___ field4;

  int field1_sum() {
    int result = 0;
    for (int i = 0; i < 3; ++i) {
      result = result + field1[i];
    }
    return result;
  }

  int field3_sum() {
    int result = 0;
    for (int i = 0; i < 3; ++i) {
      result = result + field3[i];
    }
    return result;
  }
};

IKRA_TEST_CLASSNAME::Storage IKRA_TEST_CLASSNAME::storage;
