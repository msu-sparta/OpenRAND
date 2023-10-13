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

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>

extern "C" {
#include "TestU01.h"
}

#include <openrand/phillox.h>
#include <openrand/squares.h>
#include <openrand/tyche.h>

// based on: https://www.pcg-random.org/posts/how-to-test-with-testu01.html

const char* gen_name = "Phillox";
using RNG = openrand::Phillox;

uint32_t gen32() {
  static RNG rng(42, 0);
  return rng.draw<uint32_t>();
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

uint32_t gen32_rev() {
  return rev32(gen32());
}

int main() {
  // Config options for generator output
  bool reverseBits = false;

  // Name of the generator

  std::string genName = gen_name;

  printf("Testing %s:\n", genName.c_str());
  fflush(stdout);

  // Create a generator for TestU01.

  unif01_Gen* gen = unif01_CreateExternGenBits((char*)genName.c_str(),
                                               reverseBits ? gen32 : gen32_rev);

  // Run tests.

  bbattery_SmallCrush(gen);
  fflush(stdout);

  // bbattery_Crush(gen);
  // fflush(stdout);

  // bbattery_BigCrush(gen);
  // fflush(stdout);

  // Clean up.
  unif01_DeleteExternGenBits(gen);

  return 0;
}
