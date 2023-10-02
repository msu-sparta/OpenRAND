add_executable(raw_speed raw_speed.cpp)
target_include_directories(raw_speed PRIVATE ${CMAKE_SOURCE_DIR}/include)


include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA FOUND")
        enable_language(CUDA)
        add_executable(raw_speed_cuda raw_speed_cuda.cu)
        target_include_directories(raw_speed_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)
else()
        message(STATUS "CUDA not available")
endif()