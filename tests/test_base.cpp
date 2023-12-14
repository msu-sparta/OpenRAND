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

#include <gtest/gtest.h>
#include <openrand/philox.h>
#include <openrand/squares.h>
#include <openrand/threefry.h>
#include <openrand/tyche.h>

#include <random>

template <typename RNG>
void test_rangev2(int seed) {
  RNG rng(seed, 0);
  for (int i = 0; i < 10; i++) {
    ASSERT_LT(rng.range(10), 10);
  }
  const int v = (1 << 27);
  for (int i = 0; i < 10; i++) {
    // had to create tmp variable. Couldn't directly pass to ASSERT_LT for
    // some reason
    auto x = rng.template range<true, int>(10);
    ASSERT_LT(x, 10);
    auto y = rng.template range<true, int>(v);
    ASSERT_LT(y, v);
  }
  for (int i = 0; i < 10; i++) {
    auto z = rng.template range<true, short>(1000);
    ASSERT_LT(z, 1000);
  }
}

TEST(BASE, rangev2) {
  test_rangev2<openrand::Philox>(42);
  test_rangev2<openrand::Tyche>(37);
  test_rangev2<openrand::Philox>(12345);
  test_rangev2<openrand::Tyche>(1234);
}