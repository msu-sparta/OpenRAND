#ifndef BASE_STATE_H
#define BASE_STATE_H

#include <cstdint>
#include <limits>
#include <type_traits>

#include "util.h"

/**
 * @brief Base class for random number generators.
 * 
 * This class utilizes the CRTP pattern to inject some common functionalities
 * into random number generators.
 * 
 * @tparam RNG Random number generator class. It must contain two public methods,
 * a constructor and a templated draw() function that returns the next 32 or 64
 * bit random unsigned int from its stream.
*/
template <typename RNG> class BaseRNG {
public:

  /**
   * @brief Generates a random number from a uniform distribution between 0 and 1.
   * 
   * @note Some generators may expose a more efficient version of this function that
   * returns multiple values at once.
   * 
   * @tparam T Data type to be returned. Can be 32 or 64 bit integer, float or double.
   * @return T random number from a uniform distribution between 0 and 1
  */
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


  /**
   * @brief Generates a random number from a normal distribution with mean 0 and std 1.
   * 
   * This function implements box-muller method. This method avoids branching,
   * and therefore more efficient on GPU. see:
   * https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application 
   * 
   * @tparam T floating point type to be returned
   * @return T random number from a normal distribution with mean 0 and std 1
   */
  template <typename T = float> DEVICE T randn() {
    static_assert(std::is_floating_point_v<T>);
    constexpr T M_PI2 = 2 * M_PI;

    T u = rand<T>();
    T v = rand<T>();
    T r = rnd::sqrt(T(-2.0) * rnd::log(u));
    T theta = v * M_PI2;
    return r * rnd::cos(theta);
  }

  /**
   * @brief More efficient version of @ref randn, returns two values at once.
   * 
   * @tparam T floating point type to be returned
   * @return T random number from a normal distribution with mean 0 and std 1
   */
  template <typename T = float> 
  DEVICE rnd::vec2<T> randn2() {
    // Implements box-muller method
    static_assert(std::is_floating_point_v<T>);
    constexpr T M_PI2 = 2 * M_PI;

    T u = rand<T>();
    T v = rand<T>();
    T r = rnd::sqrt(T(-2.0) * rnd::log(u));
    T theta = v * M_PI2;
    return {r * rnd::cos(theta), r * rnd::sin(theta)};
  }

  /**
   * @brief Generates a random number from a normal distribution with mean and std.
   * 
   * @tparam T floating point type to be returned
   * @param mean mean of the normal distribution
   * @param std_dev standard deviation of the normal distribution
   * @return T random number from a normal distribution with mean and std
  */
  template <typename T = float>
  DEVICE T randn(const T mean, const T std_dev) {
    return mean + randn<T>() * std_dev;
  }



  /**
   * @brief Generates a random number from a gamma distribution with shape alpha and scale b.
   * 
   * Adapted from the following implementation:
   * https://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/
   * 
   * @tparam T floating point type to be returned
   * @param alpha shape parameter of the gamma distribution
   * @param b scale parameter of the gamma distribution
   * @return T random number from a gamma distribution with shape alpha and scale b
   */
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
  /**
   * @brief Returns a reference to the random number generator.
  */
  DEVICE __inline__ RNG &gen() { return *static_cast<RNG *>(this); }
};

#endif