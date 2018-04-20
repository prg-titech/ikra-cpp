class IKRA_TEST_CLASSNAME : public SoaLayout<IKRA_TEST_CLASSNAME,
                                             kClassMaxInst,
                                             IKRA_TEST_ADDRESS_MODE,
                                             StaticStorage,
                                             IKRA_TEST_LAYOUT_MODE> {
 public:
  IKRA_INITIALIZE_CLASS

  int_ field0;
  double_ field1;
  bool_ field2;
  int_ field4;

  IKRA_TEST_CLASSNAME() {}

  IKRA_TEST_CLASSNAME(int field0_a, int field4_a) : field4(field4_a) {
    field0 = field0_a;
  }

  void add_field0_to_field4() {
    field4 = field4 + field0;
  }

  int get_field4_if_field2() {
    if (field2) {
      return field4;
    } else {
      return -1;
    }
  }
};
