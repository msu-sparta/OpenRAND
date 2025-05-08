add_executable(raw_speed_cpu raw_speed_cpu.cpp)
target_link_libraries(raw_speed_cpu PRIVATE benchmark::benchmark ${PROJECT_NAME})

# CUDA-dependent benchmarks
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsycl" HAS_SYCL_FLAG)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER AND NOT HAS_SYCL_FLAG)
    message(STATUS "CUDA FOUND: Building CUDA benchmarks")
    enable_language(CUDA)
    add_executable(raw_speed_cuda raw_speed_cuda.cu)
    target_link_libraries(raw_speed_cuda PRIVATE ${PROJECT_NAME})
else()
    message(STATUS "CUDA not found: Skipping CUDA benchmarks")
endif()

# SYCL-dependent benchmarks
if (HAS_SYCL_FLAG)
    message(STATUS "SYCL FOUND: Building SYCL benchmarks")

    add_executable(raw_speed_sycl raw_speed_sycl.cpp)
    target_compile_options(raw_speed_sycl PRIVATE -fsycl)
    target_link_libraries(raw_speed_sycl PRIVATE ${PROJECT_NAME})
else()
    message(STATUS "SYCL not found or not supported by this compiler: Skipping SYCL benchmarks")
endif()
