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

#include <gtest/gtest.h>

#include <openrand/phillox.h>
#include <openrand/squares.h>
#include <openrand/threefry.h>
#include <openrand/tyche.h>

// TODO: Add std::is_trivially_copyable and std::is_trivially_destructible tests
// for all generator types

template <typename RNG>
void test_basic() {
  RNG rng(42, 0);
  EXPECT_NE(rng.template rand<long long int>(),
            rng.template rand<long long int>());

  RNG rng1(1, 0);
  RNG rng2(1, 0);
  EXPECT_EQ(rng1.template rand<int>(), rng2.template rand<int>());
  EXPECT_EQ(rng1.template rand<int>(), rng2.template rand<int>());

  RNG rng3(3, 0);
  RNG rng4(4, 0);
  EXPECT_NE(rng3.template rand<int>(),
            rng4.template rand<int>());  // this "could" happen :)
}

TEST(RNG, basic) {
  test_basic<openrand::Phillox>();
  test_basic<openrand::Tyche>();
  test_basic<openrand::Threefry>();
  test_basic<openrand::Squares>();
}

template <typename RNG>
void test_mean() {
  RNG rng(0, 0);
  int num_draws = 1000;

  float mean = 0;
  for (int i = 0; i < num_draws; i++) {
    mean += rng.template rand<float>();
  }
  mean /= num_draws;
  EXPECT_NEAR(mean, 0.5, 0.0103);  // 99.99 % confidence
}

TEST(Uniform, mean) {
  test_mean<openrand::Phillox>();
  test_mean<openrand::Tyche>();
  test_mean<openrand::Threefry>();
  test_mean<openrand::Squares>();
}
