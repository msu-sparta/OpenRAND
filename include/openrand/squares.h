// @HEADER
// *******************************************************************************
//                                OpenRAND                                       *
//   A Performance Portable, Reproducible Random Number Generation Library       *
//                                                                               *
// Copyright (c) 2023, Michigan State University                                 *
//                                                                               *
// Permission is hereby granted, free of charge, to any person obtaining a copy  *
// of this software and associated documentation files (the "Software"), to deal *
// in the Software without restriction, including without limitation the rights  *
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     *
// copies of the Software, and to permit persons to whom the Software is         *
// furnished to do so, subject to the following conditions:                      *
//                                                                               *
// The above copyright notice and this permission notice shall be included in    *
// all copies or substantial portions of the Software.                           *
//                                                                               *
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   *
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, *
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE *
// SOFTWARE.                                                                     *
//********************************************************************************
// @HEADER

#ifndef OPENRAND_SQUARES_H_
#define OPENRAND_SQUARES_H_

#include <openrand/base_state.h>
#include <stdint.h>

namespace {

#define K 0xc58efd154ce32f6d

/*
In Squares, since the seed (i.e. key) has to have certain properties to
generate good random numbers, we can't allow user to set arbritray seed.
To avoid the pitfall of weak user supplied seed, we need to hash that seed.
*/
// inline DEVICE uint64_t hash_seed(uint64_t seed){
//     return (seed + 1) * K;
// }

// inline unsigned nthset(uint16_t x, unsigned n) {
//     return _tzcnt_u64(_pdep_u64(1U << n, x));
// }

DEVICE unsigned int get_digit(unsigned int i, char *mask) {
  unsigned int j = 0;
  while (1) {
    while (mask[j] == 0) j++;
    if (i == 0) return j;
    i--;
    j++;
  }
}

DEVICE uint64_t hash_seed(int i) {
  constexpr uint64_t factorials[16] = {
      1,         1,          2,           6,
      24,        120,        720,         5040,
      40320,     362880,     3628800,     39916800,
      479001600, 6227020800, 87178291200, 1307674368000};
  uint64_t base = (1ULL << 60);

  // NOTE: For CPUs with Bit Manipulation Instruction Set 2 support, we should
  // use the commented out uint16_t mask and nthset function.

  // uint16_t mask = 0b1111'1111'1111'1111;
  char mask[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  uint64_t res = 0;
  int n = 15;
  while (n) {
    const unsigned int v = factorials[n];
    const unsigned int d = get_digit(i / v, mask);
    // const unsigned int d = nthset(mask, i / v);
    mask[d] = 0;
    // mask = mask & ~(1<<d);

    i = i % v;
    res += d * base;
    base = base >> 4;
    n--;
  }
  // unsigned int d = nthset(mask, 0);
  unsigned int d = get_digit(0, mask);
  res += d;
  return res;
}
}  // namespace

namespace openrand {

class Squares : public BaseRNG<Squares> {
 public:
  DEVICE Squares(uint64_t seed, uint32_t _ctr,
                 uint32_t global_seed = openrand::DEFAULT_GLOBAL_SEED)
      : seed(hash_seed(seed) ^ global_seed),
        ctr(static_cast<uint64_t>(_ctr) << 32) {
  }

  template <typename T = uint32_t>
  DEVICE T draw() {
    auto round = [](uint64_t x, uint64_t w) {
      x = x * x + w;
      return (x >> 32) | (x << 32);
    };
    ctr++;
    uint64_t x = ctr * seed;
    uint64_t y = x;
    uint64_t z = y + seed;

    if constexpr (std::is_same_v<T, uint32_t>) {
      x = round(x, y);
      x = round(x, z);
      x = round(x, y);
      return (x * x + z) >> 32;
    } else {
      x = round(x, y);
      x = round(x, z);
      x = round(x, y);
      x = round(x, z);
      return x ^ ((x * x + y) >> 32);
    }
  }

 private:
  const uint64_t seed;
  uint64_t ctr = 0;
};  // class Squares

}  // namespace openrand

#endif  // OPENRAND_SQUARES_H_
