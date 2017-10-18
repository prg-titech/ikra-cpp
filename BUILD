cc_library(
  name = "soa",
  hdrs = ["soa/soa.h"],
)

cc_test(
  name = "soa_test",
  srcs = [
#    "test/soa_test.cc",
    "test/soa/simple_class_test.cc",
  ],
  copts = ["-Iexternal/gtest/include"],
  deps = [
    "@googletest//:gtest_main",
    ":soa",
  ],
)

cc_binary(
  name = "codegen_test",
  srcs = ["test/soa/benchmarks/codegen_test.cc"],
  deps = [":soa"],
)
