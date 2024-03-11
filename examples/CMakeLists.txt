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

# MPI examples
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

