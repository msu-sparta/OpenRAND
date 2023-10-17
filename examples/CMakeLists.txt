# Add the executable
add_executable(basic_usage basic_usage.cpp)
add_executable(pi_openmp pi_openmp.cpp)


include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA FOUND: building CUDA examples")
        enable_language(CUDA)
        add_executable(pi_cuda pi_cuda.cu)
        target_include_directories(pi_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)
        set_property(TARGET pi_cuda PROPERTY CUDA_STANDARD 17)

else()
        message(STATUS "skipping: CUDA examples, CUDA not found")
endif()

