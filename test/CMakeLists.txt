add_executable(basic_class_test soa/basic_class_test.cc)
target_link_libraries(basic_class_test gtest gtest_main ikra)
add_test(basic_class_test basic_class_test)

add_executable(array_test soa/array_test.cc)
target_link_libraries(array_test gtest gtest_main ikra)
add_test(array_test array_test)

add_executable(pointer_arithmetics_test soa/pointer_arithmetics_test.cc)
target_link_libraries(pointer_arithmetics_test gtest gtest_main ikra)
add_test(pointer_arithmetics_test pointer_arithmetics_test)

