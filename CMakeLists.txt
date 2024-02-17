# TODO: Lower the min version required
cmake_minimum_required(VERSION 3.18)

project(OpenRAND VERSION 1.0)

# Set the C++ standard and ensure it's strictly required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the main library as an interface
add_library(${PROJECT_NAME} INTERFACE)

# Specify the C++ flags using target_compile_options and generator expressions for better control and compatibility
target_compile_options(${PROJECT_NAME} INTERFACE
    $<$<CXX_COMPILER_ID:GNU>:
        -Wall -Wextra -Wdouble-promotion -Wconversion -O3>
    $<$<CXX_COMPILER_ID:Clang>:
        -Wall -Wextra -Wconversion -O3>
    $<$<CXX_COMPILER_ID:MSVC>:
        /W4 /O2>
)

# Include directories for the interface target
target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Options to control the build process
option(OpenRAND_ENABLE_TESTS "Enable building of tests" OFF)
option(OpenRAND_ENABLE_EXAMPLES "Enable building of examples" OFF)
option(OpenRAND_ENABLE_BENCHMARKS "Enable building of benchmarks" OFF)

message(STATUS "OpenRAND_ENABLE_TESTS: ${OpenRAND_ENABLE_TESTS}")
message(STATUS "OpenRAND_ENABLE_EXAMPLES: ${OpenRAND_ENABLE_EXAMPLES}")
message(STATUS "OpenRAND_ENABLE_BENCHMARKS: ${OpenRAND_ENABLE_BENCHMARKS}")

if(OpenRAND_ENABLE_TESTS)
  # Fetch GoogleTest for managing tests
  include(FetchContent)

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE 
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE) # Prevent GTest from being installed
  FetchContent_MakeAvailable(googletest)
  
  enable_testing()
  add_subdirectory(tests)
endif()

if(OpenRAND_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if(OpenRAND_ENABLE_BENCHMARKS)
  # Fetch GoogleBenchmarks for managing benchmarks
  include(FetchContent)
  FetchContent_Declare(
      google_benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        main
      CMAKE_ARGS     -DCMAKE_BUILD_TYPE=Release  # TODO: it's still being built as DEBUG
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE) # Prevent GoogleBenchmarks from being installed
  FetchContent_MakeAvailable(google_benchmark)
  
  add_subdirectory(benchmarks)
endif()

# Installation rules for the library
include(GNUInstallDirs)

# Installation rules for the library
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Exporting and installing the package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

# Install the include directory
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/openrand
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
