#ifndef BASE_STATE_H
#define BASE_STATE_H

#include <cstdint>
#include <limits>
#include <type_traits>

#include "util.h"

/*
RNG is the random number generator class. It must contain two public methods,
a constructor and a templated draw() function that returns the next 32 or 64
bit random unsigned int from its stream.
*/
template <typename RNG> class BaseRNG {
public:
  template <typename T = float> 
  DEVICE T rand() {
    if constexpr (sizeof(T) <= 4){
      const uint32_t x = gen().template draw<uint32_t>();
      if constexpr (std::is_integral_v<T>) 
        return static_cast<T>(x);
      else 
        return uniform<float, uint32_t>(x);
    }
    else{
      const uint64_t x = gen().template draw<uint64_t>();
      if constexpr (std::is_integral_v<T>) 
        return static_cast<T>(x);
      else 
        return uniform<double, uint64_t>(x);
    }
  }

  template <typename T = float> DEVICE void fill_random(T *array, const int N) {
    for (int i = 0; i < N; i++)
      array[i] = rand<T>();
  }

  template <typename Ftype, typename Utype>
  inline DEVICE Ftype uniform(const Utype in) const {
    constexpr Ftype factor = Ftype(1.)/(Ftype(~static_cast<Utype>(0)) + Ftype(1.));
    constexpr Ftype halffactor = Ftype(0.5)*factor;
    return Utype(in)*factor + halffactor;
  }

  template <typename T = float> DEVICE T randn() {
    // Implements box-muller method
    // TODO: we can generate two numbers here instead of one
    static_assert(std::is_floating_point_v<T>);
    constexpr T M_PI2 = 2 * M_PI;

    T u = rand<T>();
    T v = rand<T>();
    T r = rnd::sqrt(T(-2.0) * rnd::log(u));
    T theta = v * M_PI2;
    return r * rnd::cos(theta);
  }

  // More efficient version of randn(), generates 2 floating point numbers at once
  template <typename T = float> 
  DEVICE rnd::vec2<T> randn2() {
    // Implements box-muller method
    // TODO: we can generate two numbers here instead of one
    static_assert(std::is_floating_point_v<T>);
    constexpr T M_PI2 = 2 * M_PI;

    T u = rand<T>();
    T v = rand<T>();
    T r = rnd::sqrt(T(-2.0) * rnd::log(u));
    T theta = v * M_PI2;
    return {r * rnd::cos(theta), r * rnd::sin(theta)};
  }

  template <typename T = float>
  DEVICE T randn(const T mean, const T std_dev) {
    return mean + randn<T>() * std_dev;
  }

  // https://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/
  template<typename T=float>
  DEVICE inline T gamma(T alpha, T b){
      T d = alpha - T((1./3.));
      T c = T(1.) / rnd::sqrt(9.f * d);
      T v, x;
      while(true){
          do{
              x = randn<T>();
              v = T(1.0) + c * x;
          }
          while (v <= T(0.));
          v = v*v*v;
          T u = rand<T>();

          const T x2 = x*x;
          if (u < 1.0f - 0.0331f * x2 *x2) 
            return (d * v * b);

          if (rnd::log(u) < 0.5f * x2 + d * (1.0f - v + rnd::log(v))) 
              return (d * v * b);
      }
    }

private:
  DEVICE __inline__ RNG &gen() { return *static_cast<RNG *>(this); }
};

#endif