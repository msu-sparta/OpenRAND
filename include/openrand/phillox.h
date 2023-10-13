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

#ifndef OPENRAND_PHILLOX_H_
#define OPENRAND_PHILLOX_H_

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

#include <openrand/base_state.h>

namespace {

constexpr uint32_t PHILOX_M4x32_0 = 0xD2511F53;
constexpr uint32_t PHILOX_M4x32_1 = 0xCD9E8D57;
constexpr uint32_t PHILOX_W32_0 = 0x9E3779B9;
constexpr uint32_t PHILOX_W32_1 = 0xBB67AE85;

inline DEVICE uint32_t mulhilo(uint32_t L, uint32_t R, uint32_t *hip) {
  uint64_t product = static_cast<uint64_t>(L) * static_cast<uint64_t>(R);
  *hip = product >> 32;
  return static_cast<uint32_t>(product);
}

inline DEVICE void round(const uint32_t (&key)[2], uint32_t (&ctr)[4]) {
  uint32_t hi0;
  uint32_t hi1;
  uint32_t lo0 = mulhilo(PHILOX_M4x32_0, ctr[0], &hi0);
  uint32_t lo1 = mulhilo(PHILOX_M4x32_1, ctr[2], &hi1);
  ctr[0] = hi1 ^ ctr[1] ^ key[0];
  ctr[1] = lo1;
  ctr[2] = hi0 ^ ctr[3] ^ key[1];
  ctr[3] = lo0;
}
}  // namespace

namespace openrand {

/**
 * @class Phillox
 * @brief Phillox generator
 * @note This is a modified version of Phillox generator from Random123 library.
 */
class Phillox : public BaseRNG<Phillox> {
 public:
  DEVICE Phillox(uint64_t seed, uint32_t ctr,
                 uint32_t global_seed = openrand::DEFAULT_GLOBAL_SEED)
      : seed_hi((uint32_t)(seed >> 32)),
        seed_lo((uint32_t)(seed & 0xFFFFFFFF)),
        initctr_hi(global_seed),
        initctr_lo(ctr) {
  }

  template <typename T = uint32_t>
  DEVICE T draw() {
    generate();

    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
    if constexpr (std::is_same_v<T, uint32_t>) return _out[0];

    uint64_t res =
        (static_cast<uint64_t>(_out[0]) << 32) | static_cast<uint64_t>(_out[1]);
    return static_cast<uint64_t>(res);
  }

  openrand::uint4 draw_int4() {
    generate();
    return openrand::uint4{_out[0], _out[1], _out[2], _out[3]};
  }

  openrand::float4 draw_float4() {
    generate();
    return openrand::float4{
        uniform<float, uint32_t>(_out[0]), uniform<float, uint32_t>(_out[1]),
        uniform<float, uint32_t>(_out[2]), uniform<float, uint32_t>(_out[3])};
  }

 private:
  DEVICE void generate() {
    uint32_t key[2] = {seed_hi, seed_lo};
    // The counter takes one of the 4 values from internal counter, one from
    // global seed, and one is what the user provided during instantiation.
    // Another is left constant.

    // The internal counter helps to avoid forcing user to increment counter
    // each time a number is generated.

    _out[0] = 0x12345;
    _out[1] = _ctr;
    _out[2] = initctr_hi;
    _out[3] = initctr_lo;

    for (int r = 0; r < 10; r++) {
      if (r > 0) {
        key[0] += PHILOX_W32_0;
        key[1] += PHILOX_W32_1;
      }
      round(key, _out);
    }
    _ctr++;
  }

  // User provided seed and counter broken up, constant throughout
  // the lifetime of the object
  const uint32_t seed_hi, seed_lo;
  const uint32_t initctr_hi, initctr_lo;
  // private counter to keep track of numbers generated by this instance of rng
  uint32_t _ctr = 0;
  uint32_t _out[4];
};  // class Phillox

}  // namespace openrand

#endif  // OPENRAND_PHILLOX_H_
