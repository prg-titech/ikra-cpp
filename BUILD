cc_library(
  name = "soa",
  hdrs = [
    "soa/array_field.h",
    "soa/constants.h",
    "soa/field.h",
    "soa/layout.h",
    "soa/soa.h",
  ],
  textual_hdrs = [
    "soa/class_initialization.def",
    "soa/storage.def",
  ],
  deps = [":executor"],
)

cc_library(
  name = "executor",
  hdrs = [
    "executor/array.h",
    "executor/iterator.h",
  ],
)

cc_library(
  name = "soa_test_headers",
  textual_hdrs = [
    "test/soa/basic_class_test_layout.def",
  ],
)

cc_test(
  name = "soa_test",
  srcs = [
#    "test/soa/array_test.cc",
    "test/soa/basic_class_test.cc",
    "test/soa/pointer_arithmetics_test.cc",
  ],
  copts = ["-Iexternal/gtest/include"],
  deps = [
    "@googletest//:gtest_main",
    ":soa",
    ":soa_test_headers",
  ],
)

cc_test(
  name = "executor_test",
  srcs = ["test/executor_test.cc"],
  deps = [
    "@googletest//:gtest_main",
    ":soa",
    ":executor"
  ],
)

cc_binary(
  name = "codegen_test",
  srcs = ["test/soa/benchmarks/codegen_test.cc"],
  deps = [":soa"],
)
