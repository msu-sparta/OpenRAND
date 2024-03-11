find_package(Python COMPONENTS Interpreter REQUIRED)

enable_testing()

add_executable(
  uniform
  test_uniform.cpp
)
target_link_libraries(
  uniform PRIVATE GTest::gtest_main ${PROJECT_NAME}
)

add_executable(
  normal
  test_normal.cpp
)
target_link_libraries(
  normal PRIVATE GTest::gtest_main ${PROJECT_NAME}
)

add_executable(
  base
  test_base.cpp
)
target_link_libraries(
  base PRIVATE GTest::gtest_main ${PROJECT_NAME}
)

include(GoogleTest)
gtest_discover_tests(uniform)
gtest_discover_tests(normal)
gtest_discover_tests(base)


# Statistical tests, not run through gtest framework
add_executable(pract_rand pract_rand.cpp)
add_executable(pract_rand_multi pract_rand_multistream.cpp)
target_link_libraries(pract_rand ${PROJECT_NAME})
target_link_libraries(pract_rand_multi ${PROJECT_NAME})

OPTION(TESTU01_PATH "Path to the TestU01 library" "")
MESSAGE(STATUS "TESTU01_PATH: ${TESTU01_PATH}")
if(TESTU01_PATH)
    message(STATUS "Building TESTU01 statistical test suite")
    link_directories(${TESTU01_PATH}/lib)
    add_executable(testu01_serial testu01.cpp)
    target_link_libraries(testu01_serial testu01 probdist mylib ${PROJECT_NAME})
    target_include_directories(testu01_serial PUBLIC ${TESTU01_PATH}/include)

    add_executable(testu01_multi testu01-multistream.cpp)
    target_link_libraries(testu01_multi testu01 probdist mylib ${PROJECT_NAME})
    target_include_directories(testu01_multi PUBLIC ${TESTU01_PATH}/include)
else()
    message(STATUS "TESTU01_PATH not set, not building statistical test suite")
endif()