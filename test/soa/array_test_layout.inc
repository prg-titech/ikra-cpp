class IKRA_TEST_CLASSNAME
    : public SoaLayout<IKRA_TEST_CLASSNAME, kClassMaxInst,
                       IKRA_TEST_ADDRESS_MODE> {
 public:
  IKRA_INITIALIZE_CLASS

  int_ field0;

  // Array has size 12 bytes.
  array_(int, 3, object) field1;
  int_ field2;

  // Array has size 12 bytes.
  array_(int, 3, fully_inlined) field3;
  int_ field4;

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

