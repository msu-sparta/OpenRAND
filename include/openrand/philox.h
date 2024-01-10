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

#ifndef OPENRAND_Philox_H_
#define OPENRAND_Philox_H_

#include <openrand/base_state.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

#define PHILOX_W0 0x9E3779B9
#define PHILOX_W1 0xBB67AE85
#define PHILOX_M0 0xD2511F53
#define PHILOX_M1 0xCD9E8D57

namespace openrand {

/**
 * @class Philox
 * @brief Philox generator
 * @note This is a modified version of Philox generator from Random123 library.
 * This uses 4x 32-bit counter, 2x 32-bit key along with 10 rounds.
 */
class Philox : public BaseRNG<Philox> {
 public:
  /**
   * @brief Construct a new Philox generator
   *
   * @note Internally, global_seed is treated in the same way as other counters,
   * and can be treated as such depending on the application needs.
   *
   * @param seed 64-bit seed
   * @param ctr 32-bit counter
   * @param global_seed (Optional) 32-bit global seed.
   * @param ctr1 (Optional) Another 32-bit counter exposed for advanced use.
   */
  DEVICE Philox(uint64_t seed, uint32_t ctr,
                uint32_t global_seed = openrand::DEFAULT_GLOBAL_SEED,
                uint32_t ctr1 = 0x12345)
      : seed_hi((uint32_t)(seed >> 32)),
        seed_lo((uint32_t)(seed & 0xFFFFFFFF)),
        ctr0(ctr),
        ctr1(ctr1),
        ctr2(global_seed) {
  }

  template <typename T = uint32_t>
  DEVICE T draw() {
    generate();

    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
    if constexpr (std::is_same_v<T, uint32_t>) return _out[0];

    // Not wrapping this block in else{} would lead to compiler warning
    else {
      uint64_t res = (static_cast<uint64_t>(_out[0]) << 32) |
                     static_cast<uint64_t>(_out[1]);
      return static_cast<uint64_t>(res);
    }
  }

  openrand::uint4 draw_int4() {
    generate();
    return openrand::uint4{_out[0], _out[1], _out[2], _out[3]};
  }

  openrand::float4 draw_float4() {
    generate();
    return openrand::float4{
        u01<float, uint32_t>(_out[0]), u01<float, uint32_t>(_out[1]),
        u01<float, uint32_t>(_out[2]), u01<float, uint32_t>(_out[3])};
  }

 private:
  DEVICE void generate() {
    uint32_t key[2] = {seed_hi, seed_lo};
    /**
     * Philox 4x32 can take upto 4 counters. Here, one counter is part of the
     * general API, mandatory during instantiation. One is (optional) global seed.
     * Third one can be optionally set by user. 4th one is interanally managed.
     *
     * The internal counter helps to avoid forcing user to increment counter
     * each time a number is generated.
     */

    _out[0] = ctr0;
    _out[1] = ctr1;
    _out[2] = ctr2;
    _out[3] = _ctr;

    for (int r = 0; r < 10; r++) {
      if (r > 0) {
        key[0] += PHILOX_W0;
        key[1] += PHILOX_W1;
      }
      round(key, _out);
    }
    _ctr++;
  }

  inline DEVICE uint32_t mulhilo(uint32_t L, uint32_t R, uint32_t *hip) {
    uint64_t product = static_cast<uint64_t>(L) * static_cast<uint64_t>(R);
    *hip = static_cast<uint32_t>(product >> 32);
    return static_cast<uint32_t>(product);
  }

  inline DEVICE void round(const uint32_t (&key)[2], uint32_t (&ctr)[4]) {
    uint32_t hi0;
    uint32_t hi1;
    uint32_t lo0 = mulhilo(PHILOX_M0, ctr[0], &hi0);
    uint32_t lo1 = mulhilo(PHILOX_M1, ctr[2], &hi1);
    ctr[0] = hi1 ^ ctr[1] ^ key[0];
    ctr[1] = lo1;
    ctr[2] = hi0 ^ ctr[3] ^ key[1];
    ctr[3] = lo0;
  }

  // User provided seed and counter, constant throughout
  // the lifetime of the rng object
  const uint32_t seed_hi, seed_lo;
  const uint32_t ctr0, ctr1, ctr2;
  uint32_t _out[4];

 public:
  // internal counter to keep track of numbers generated by this instance of rng
  uint32_t _ctr = 0;
};  // class Philox

}  // namespace openrand

#endif  // OPENRAND_Philox_H_
