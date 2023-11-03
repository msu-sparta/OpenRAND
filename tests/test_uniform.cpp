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
#include <openrand/phillox.h>
#include <openrand/squares.h>
#include <openrand/threefry.h>
#include <openrand/tyche.h>

#include <random>

// std::is_trivially_copyable and std::is_trivially_destructible tests
// for all generator types
TEST(RNG, trivially_copyable) {
  EXPECT_TRUE(std::is_trivially_copyable<openrand::Phillox>::value);
  EXPECT_TRUE(std::is_trivially_copyable<openrand::Tyche>::value);
  EXPECT_TRUE(std::is_trivially_copyable<openrand::Threefry>::value);
  EXPECT_TRUE(std::is_trivially_copyable<openrand::Squares>::value);
}

TEST(RNG, trivially_destructible) {
  EXPECT_TRUE(std::is_trivially_destructible<openrand::Phillox>::value);
  EXPECT_TRUE(std::is_trivially_destructible<openrand::Tyche>::value);
  EXPECT_TRUE(std::is_trivially_destructible<openrand::Threefry>::value);
  EXPECT_TRUE(std::is_trivially_destructible<openrand::Squares>::value);
}

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
void test_range(){
  RNG rng (1234567, 1234567);

  double mean = 0;
  for(int i = 0; i < 1000; i++){
    auto x = rng.uniform(10.0, 20.0);
    mean += x;
    EXPECT_TRUE((x >= 10.0) && (x <= 20.0));
  }
  EXPECT_NEAR(mean / 1000.0, 15.0, 0.36);  // 99.99 % confidence

  mean = 0;
  for(int i = 0; i < 1000; i++){
    auto x = rng.uniform(-20.0, -10.0);
    mean += x;
    EXPECT_TRUE((x >= -20.0) && (x <= -10.0));
  }
  EXPECT_NEAR(mean / 1000.0, -15.0, 0.36);

  // For integer type, this method is slightly biased towards lower numbers.
  mean = 0;
  for(int i = 0; i < 1000; i++){
    auto x = rng.template uniform<int>(10,20);
    mean += (float)x;
    EXPECT_TRUE((x >= 10) && (x <= 20));
  }
  EXPECT_NEAR(mean / 1000.0, 15.0, 1.0);
}

TEST(RNG, range){
  test_range<openrand::Phillox>();
  test_range<openrand::Tyche>();
  test_range<openrand::Threefry>();
  test_range<openrand::Squares>();
}

template <typename RNG>
void test_mean() {
  RNG rng(0, 0);
  int num_draws = 1000;
  std::uniform_real_distribution<float> rdist(0, 1.0);

  float mean = 0;
  for (int i = 0; i < num_draws; i++) {
    mean += rdist(rng);
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

template <typename RNG>
void test_cpp_engine() {
  RNG rng(42, 0);

  std::uniform_int_distribution<int> udist(0, 100);
  udist(rng);

  std::uniform_real_distribution<float> rdist(0, 100);
  rdist(rng);

  std::normal_distribution<float> ndist(0, 10.0f);
  ndist(rng);

  std::bernoulli_distribution bd(0.25);
  bd(rng);

  std::lognormal_distribution<double> lnd(1.6, 0.25);
  lnd(rng);

  std::student_t_distribution<> student_d{10.0f};
  student_d(rng);
}

TEST(CPP11, engine) {
  test_cpp_engine<openrand::Phillox>();
  test_cpp_engine<openrand::Tyche>();
  test_cpp_engine<openrand::Threefry>();
  test_cpp_engine<openrand::Squares>();
}
