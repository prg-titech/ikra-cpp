cc_library(
  name = "sdl2",
  hdrs = glob(["include/SDL2/*.h"]),
  defines = ["_REENTRANT"],
  includes = ["include/SDL2"],
  linkopts = ["-lSDL2"],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "soa",
  hdrs = [
    "soa/array_field.h",
    "soa/constants.h",
    "soa/field.h",
    "soa/field_type_generator.h",
    "soa/layout.h",
    "soa/preprocessor.h",
    "soa/soa.h",
    "soa/storage.h",
  ],
  textual_hdrs = [
    "soa/class_initialization.inc",
  ],
  deps = [":executor"],
)

cc_library(
  name = "executor",
  hdrs = [
    "executor/executor.h",
    "executor/iterator.h",
  ],
)

cc_library(
  name = "soa_test_headers",
  textual_hdrs = [
    "test/soa/array_test_layout.inc",
    "test/soa/basic_class_test_layout.inc",
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
    ":soa_test_headers",
  ],
)

cc_library(
  name = "executor_test_headers",
  textual_hdrs = [
    "test/executor_test_layout.inc",
  ],
)

cc_test(
  name = "executor_test",
  srcs = ["test/executor_test.cc"],
  deps = [
    "@googletest//:gtest_main",
    ":soa",
    ":executor",
    ":executor_test_headers",
  ],
)

cc_binary(
  name = "particle_simulation",
  srcs = ["example/particle_simulation.cc"],
  deps = [
    ":sdl2",
    ":soa",
    ":executor",
  ],
)

cc_library(
  name = "bfs_util",
  hdrs = ["example/bfs/bfs_loader.h"],
  deps = [":soa"],
)

cc_binary(
  name = "bfs_const_size_array",
  srcs = ["example/bfs/bfs_const_size_array.cc"],
  deps = [
    ":bfs_util",
    ":soa",
    ":executor",
  ]
)

cc_binary(
  name = "bfs_std_vector",
  srcs = ["example/bfs/bfs_std_vector.cc"],
  deps = [
    ":bfs_util",
    ":soa",
    ":executor",
  ]
)
