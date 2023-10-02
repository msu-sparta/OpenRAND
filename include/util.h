#ifndef UTIL_H
#define UTIL_H

#include <cmath>
#include <cstdint>

#ifdef __CUDA_ARCH__
  #define DEVICE __host__ __device__
#elif defined(__HIP_DEVICE_COMPILE__)
  #define DEVICE __device__ __host__
#else
  #define DEVICE 
#endif


namespace rnd{

// NOTE: nvcc compiler replaces floating point variants with cuda built-in versions
// NOTE: floating point variants are not part of std namespace for some reason

constexpr uint32_t DEFAULT_GLOBAL_SEED = 0xAAAAAAAA;  // equal number of 0 and 1 bits

template <typename T> 
inline DEVICE T sin(T x) {
    if constexpr (std::is_same_v<T, float>)
        return sinf(x);
    else if constexpr (std::is_same_v<T, double>)
        return std::sin(x);
}

template <typename T>
inline DEVICE T cos(T x) {
    if constexpr (std::is_same_v<T, float>)
        return cosf(x);
    else if constexpr (std::is_same_v<T, double>)
        return std::cos(x);
}

template <typename T>
inline DEVICE T log(T x) {
    if constexpr (std::is_same_v<T, float>)
        return logf(x);
    else if constexpr (std::is_same_v<T, double>)
        return std::log(x);
}

template <typename T>
inline DEVICE T sqrt(T x) {
    if constexpr (std::is_same_v<T, float>)
        return sqrtf(x);
    else if constexpr (std::is_same_v<T, double>)
        return std::sqrt(x);
}


template <typename T>
struct vec2{
    T x, y;
};

template <typename T>
struct vec3{
    T x, y, z;
};

template <typename T>
struct vec4{
    T x, y, z, w;
};

// for GPU, better to be explicit about the type
using uint2 = rnd::vec2<uint32_t>;
using uint3 = rnd::vec3<uint32_t>;
using uint4 = rnd::vec4<uint32_t>;

using float2 = rnd::vec2<float>;
using float3 = rnd::vec3<float>;
using float4 = rnd::vec4<float>;

using double2 = rnd::vec2<double>;
using double3 = rnd::vec3<double>;
using double4 = rnd::vec4<double>;

} // namespace rnd




#endif
