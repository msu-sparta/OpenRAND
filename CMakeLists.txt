# TODO: Lower the min version required
cmake_minimum_required(VERSION 3.18)

# Project's name
project(OpenRAND)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#- Wdouble-promotion : implicit float->double promotion is a really bad idea for GPU
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wdouble-promotion -Wconversion -O3 -fopenmp")

include_directories(${PROJECT_SOURCE_DIR}/include)

include(FetchContent)

# Build TestU01 statistical test suite
set(TESTU01_PATH "" CACHE PATH "Path to TestU01 library")

# Only build tests, examples and benchmarks if this isn't compiled
# as a dependecy of another project.
# Thanks to: https://www.foonathan.net/2022/06/cmake-fetchcontent/
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  enable_testing()
  add_subdirectory(tests)
  
  add_subdirectory(examples)
  add_subdirectory(benchmarks)

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE 
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  # TODO: it's still being built as DEBUG
  FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        main
    CMAKE_ARGS     -DCMAKE_BUILD_TYPE=Release
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(google_benchmark)
endif()




add_library(cbrng INTERFACE)
target_include_directories(cbrng
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)


