// @HEADER
// *******************************************************************************
//                                OpenRAND *
//   A Performance Portable, Reproducible Random Number Generation Library *
//                                                                               *
// Copyright (c) 2023, Michigan State University *
//                                                                               *
// Permission is hereby granted, free of charge, to any person obtaining a copy
// * of this software and associated documentation files (the "Software"), to
// deal * in the Software without restriction, including without limitation the
// rights  * to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell     * copies of the Software, and to permit persons to whom the
// Software is         * furnished to do so, subject to the following
// conditions:                      *
//                                                                               *
// The above copyright notice and this permission notice shall be included in *
// all copies or substantial portions of the Software. *
//                                                                               *
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, *
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE *
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE * SOFTWARE. *
//********************************************************************************
// @HEADER

/**
 * @file
 * @brief Simplified version of the multi-stream correlation test outlined in
 * [1].
 *
 * We hold either key or counter constant, and increment other by 1. This tests
 * both multi-stream and sub-streams approaches of creating mulptiple streams
 * [1]. Although simplified, this closely resembles how we expect these
 * parameters to be used in real-world parallel programs.
 *
 * Based on: https://www.pcg-random.org/posts/how-to-test-with-testu01.html
 *
 * [1] Salmon, John K., et al. "Parallel random numbers: as easy as 1, 2, 3."
 * Proceedings of 2011 international conference for high performance computing,
 * networking, storage and analysis. 2011.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

extern "C" {
#include "TestU01.h"
}

#include <openrand/philox.h>
#include <openrand/squares.h>
#include <openrand/threefry.h>
#include <openrand/tyche.h>

// Control parameters for this test program.
const int C = 3;     // C numbers from per stream
const int NS = 100;  // from NS streams
// end

std::vector<uint32_t> buffer;
int ctr = 0;

template <typename RNG>
void populate_buffer() {
  std::vector<RNG> generators;
  for (int i = 0; i < NS; i++) {
    generators.emplace_back(i, ctr);
  }

  for (int i = 0; i < NS; i++) {
    for (int j = 0; j < C; j++) {
      buffer.push_back(generators[i].template draw<uint32_t>());
    }
  }
  std::reverse(buffer.begin(), buffer.end());
}

template <typename RNG>
uint32_t gen32() {
  if (buffer.empty()) {
    populate_buffer<RNG>();
    ctr++;
  }
  uint32_t res = buffer.back();
  buffer.pop_back();
  return res;
}

// END OF GENERATOR SECTION

inline uint32_t rev32(uint32_t v) {
  // https://graphics.stanford.edu/~seander/bithacks.html
  // swap odd and even bits
  v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
  // swap consecutive pairs
  v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
  // swap nibbles ...
  v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
  // swap bytes
  v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
  // swap 2-byte-long pairs
  v = (v >> 16) | (v << 16);
  return v;
}

template <typename RNG>
uint32_t gen32_rev() {
  return rev32(gen32<RNG>());
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Please provide generator name and crush level" << std::endl;
    return 1;
  }

  std::string arg = argv[1];
  std::string crush_level = argv[2];
  bool reverseBits = false;

  std::cout << "Generating multi-stream for: " << arg << std::endl;

  unif01_Gen* gen;
  if (arg == "philox") {
    gen = unif01_CreateExternGenBits(
        (char*)arg.c_str(),
        reverseBits ? gen32<openrand::Philox> : gen32_rev<openrand::Philox>);
  } else if (arg == "tyche") {
    gen = unif01_CreateExternGenBits(
        (char*)arg.c_str(),
        reverseBits ? gen32<openrand::Tyche> : gen32_rev<openrand::Tyche>);
  } else if (arg == "squares") {
    gen = unif01_CreateExternGenBits(
        (char*)arg.c_str(),
        reverseBits ? gen32<openrand::Squares> : gen32_rev<openrand::Squares>);
  } else if (arg == "threefry") {
    gen = unif01_CreateExternGenBits(
        (char*)arg.c_str(), reverseBits ? gen32<openrand::Threefry>
                                        : gen32_rev<openrand::Threefry>);
  } else {
    std::cout << "Invalid argument." << std::endl;
    return 1;
  }

  // Run tests.

  if (crush_level == "small") {
    bbattery_SmallCrush(gen);
    fflush(stdout);
  } else if (crush_level == "crush") {
    bbattery_Crush(gen);
    fflush(stdout);
  } else if (crush_level == "big") {
    bbattery_BigCrush(gen);
    fflush(stdout);
  } else {
    std::cout << "Invalid crush level." << std::endl;
    return 1;
  }

  // Clean up.
  unif01_DeleteExternGenBits(gen);

  return 0;
}
