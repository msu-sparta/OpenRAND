#ifndef TYCHE_H
#define TYCHE_H

#include <cstdint>
#include <iostream>
#include <limits>

#include "base_state.hpp"

namespace{
  inline DEVICE uint32_t rotl(uint32_t value, unsigned int x) {
    return (value << x) | (value >> (32 - x));
  }
}

class Tyche : public BaseRNG<Tyche> {
public:
  DEVICE Tyche(uint64_t seed, uint32_t ctr, uint32_t global_seed=rnd::DEFAULT_GLOBAL_SEED) {
    seed = seed ^ global_seed;
    a = static_cast<uint32_t>(seed >> 32);
    b = static_cast<uint32_t>(seed & 0xFFFFFFFFULL);
    d = d ^ ctr;

    for (int i = 0; i < 20; i++) {
      mix();
    }
  }

  template <typename T = uint32_t> DEVICE T draw() {
    mix();
    if constexpr (std::is_same_v<T, uint32_t>)
      return a;
    
    else{
      uint32_t tmp = a;
      mix();
      uint64_t res = (static_cast<uint64_t>(tmp) << 32) | a;
      return static_cast<T>(res);
    }
  }

private:
  // inline DEVICE void mix() {
  //   a += b;
  //   d = rotl(d ^ a, 16);
  //   c += d;
  //   b = rotl(b ^ c, 12);
  //   a += b;
  //   d = rotl(d ^ a, 8);
  //   c += d;
  //   b = rotl(b ^ c, 7);
  // }

  inline DEVICE void mix(){
    b = rotl(b, 7) ^ c;
    c -= d;
    d = rotl(d, 8) ^ a;
    a -= b;
    b = rotl(b, 12) ^ c;
    c -= d;
    d = rotl(d, 16) ^ a; 
    a -= b;
  }

  uint32_t a, b;
  uint32_t c = 2654435769;
  uint32_t d = 1367130551;
};

#endif // TYCHE_H