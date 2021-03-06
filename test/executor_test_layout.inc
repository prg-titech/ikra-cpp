class IKRA_TEST_CLASSNAME : public SoaLayout<IKRA_TEST_CLASSNAME,
                                             kClassMaxInst,
                                             IKRA_TEST_ADDRESS_MODE> {
 public:
  IKRA_INITIALIZE_CLASS

  int_ field0;
  int_ field1;

  IKRA_TEST_CLASSNAME(int a, int b) : field0(a), field1(b) {}

  void add_field1_and_a_to_field0(int a) {
    field0 += field1 + a;
  }

  int sum_plus_delta(int delta) {
    return field0 + field1 + delta;
  }
};
