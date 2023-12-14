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
#include <openrand/philox.h>
#include <openrand/squares.h>
#include <openrand/threefry.h>
#include <openrand/tyche.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

double standard_normal_cdf(double x) {
  return 0.5 * std::erfc(-x * M_SQRT1_2);
}

std::pair<double, bool> ks_test(const std::vector<double>& data) {
  auto n = data.size();
  std::vector<double> data_sorted(data);
  std::sort(data_sorted.begin(), data_sorted.end());

  double max_diff = 0.0;

  for (unsigned int i = 0; i < n; ++i) {
    double cdf_data = static_cast<double>(i + 1) / static_cast<double>(n);
    double cdf_theoretical = standard_normal_cdf(data_sorted[i]);
    max_diff = std::max(max_diff, std::abs(cdf_data - cdf_theoretical));
  }

  // Calculate the critical value
  double alpha = 0.05;  // significance level
  double critical_value = std::sqrt(-0.5 * std::log(alpha / 2.0)) *
                          std::sqrt(static_cast<double>(n));

  bool reject_null = max_diff > critical_value;

  return {max_diff, reject_null};
}

template <typename RNG>
void test_normalcy() {
  RNG rng(42, 0);
  const int num_draws = 200;  // should be more than enough

  std::vector<double> sample_data;
  for (int i = 0; i < num_draws / 2; i++)
    sample_data.push_back(rng.template randn<float>());

  while (sample_data.size() < num_draws) {
    openrand::double2 d2 = rng.template randn2<double>();
    sample_data.push_back(d2.x);
    sample_data.push_back(d2.y);
  }

  auto [max_diff, reject_null] = ks_test(sample_data);

  std::cout << "Maximum difference between CDFs: " << max_diff << std::endl;
  ASSERT_TRUE(!reject_null);
}

TEST(normal, Philox) {
  test_normalcy<openrand::Philox>();
}

TEST(normal, tyche) {
  test_normalcy<openrand::Tyche>();
}

TEST(normal, threefry) {
  test_normalcy<openrand::Threefry>();
}

TEST(normal, squares) {
  test_normalcy<openrand::Squares>();
}
