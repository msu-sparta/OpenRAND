# OpenMP-dependent examples
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP FOUND: Building OpenMP examples")

    add_executable(serial_parallel_equivalence state_forwarding.cpp)
    target_link_libraries(serial_parallel_equivalence PRIVATE OpenMP::OpenMP_CXX ${PROJECT_NAME})
    
    add_executable(basic_usage basic_usage.cpp)
    target_link_libraries(basic_usage PRIVATE OpenMP::OpenMP_CXX ${PROJECT_NAME})

    add_executable(pi_openmp pi_openmp.cpp)
    target_link_libraries(pi_openmp PRIVATE OpenMP::OpenMP_CXX ${PROJECT_NAME})
else()
    message(STATUS "OpenMP not found: Skipping OpenMP examples")
endif()

# CUDA-dependent examples
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA FOUND: Building CUDA examples")
    enable_language(CUDA)

    add_executable(pi_cuda pi_cuda.cu)
    target_link_libraries(pi_cuda PRIVATE ${PROJECT_NAME})
    set_property(TARGET pi_cuda PROPERTY CUDA_STANDARD 17)
else()
    message(STATUS "CUDA not found: Skipping CUDA examples")
endif()

# SYCL-dependent examples
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsycl" HAS_SYCL_FLAG)

if (HAS_SYCL_FLAG)
    message(STATUS "SYCL FOUND: Building SYCL examples")

    add_executable(pi_sycl pi_sycl.cpp)
    target_compile_options(pi_sycl PRIVATE -fsycl)
    target_link_libraries(pi_sycl PRIVATE ${PROJECT_NAME})
else()
    message(STATUS "SYCL not found or not supported by this compiler: Skipping SYCL examples")
endif()

