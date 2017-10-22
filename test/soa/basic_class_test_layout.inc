class IKRA_TEST_CLASSNAME : public SoaLayout<IKRA_TEST_CLASSNAME,
                                             kClassMaxInst,
                                             IKRA_TEST_ADDRESS_MODE> {
 public:
  #include IKRA_INITIALIZE_CLASS

  int___ field0;
  double___ field1;
  bool___ field2;
  int___ field4;

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

IKRA_TEST_CLASSNAME::Storage IKRA_TEST_CLASSNAME::storage;
