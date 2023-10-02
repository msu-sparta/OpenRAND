find_package(Python COMPONENTS Interpreter REQUIRED)


enable_testing()

add_executable(
  uniform
  test_uniform.cpp
)
target_link_libraries(
  uniform
  GTest::gtest_main
)

add_executable(
  normal
  test_normal.cpp
)
target_link_libraries(
  normal
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(uniform)
gtest_discover_tests(normal)


# Statistical tests, not run through gtest framework
add_executable(pract_rand pract_rand.cpp)
add_executable(pract_rand_multi pract_rand_multistream.cpp)

if(DEFINED TESTU01)
    message(STATUS "Building TESTU01 statistical test suite")
    link_directories(${TESTU01_PATH}/lib)
    add_executable(testu01_serial testu01.cpp)
    target_link_libraries(testu01_serial testu01 probdist mylib)
    target_include_directories(testu01_serial PUBLIC ${TESTU01_PATH}/include)

    add_executable(testu01_multi testu01-multistream.cpp)
    target_link_libraries(testu01_multi testu01 probdist mylib)
    target_include_directories(testu01_multi PUBLIC ${TESTU01_PATH}/include)
else()
    message(STATUS "TESTU01 not set, not building statistical test suite")
endif()