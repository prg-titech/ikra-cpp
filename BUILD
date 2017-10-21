cc_library(
  name = "soa",
  hdrs = [
    "soa/array_field.h",
    "soa/soa.h",
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

cc_test(
  name = "soa_test",
  srcs = [
    "test/soa/array_test.cc",
    "test/soa/basic_class_test.cc",
    "test/soa/pointer_arithmetics_test.cc",
  ],
  copts = ["-Iexternal/gtest/include"],
  deps = [
    "@googletest//:gtest_main",
    ":soa",
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
