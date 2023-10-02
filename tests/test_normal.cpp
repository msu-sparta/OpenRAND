#include <cstdint>
#include <iostream>
#include <vector>
#include <fstream>
#include <gtest/gtest.h>

#include "phillox.h"
#include "tyche.h"
#include "threefry.h"
#include "squares.h"

using std::cout;
using std::endl;

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <functional>

double standard_normal_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

std::pair<double, bool> ks_test(const std::vector<double>& data) {
    int n = data.size();
    std::vector<double> data_sorted(data);
    std::sort(data_sorted.begin(), data_sorted.end());

    double max_diff = 0.0;

    for (int i = 0; i < n; ++i) {
        double cdf_data = static_cast<double>(i + 1) / static_cast<double>(n);
        double cdf_theoretical = standard_normal_cdf(data_sorted[i]);
        max_diff = std::max(max_diff, std::abs(cdf_data - cdf_theoretical));
    }

    // Calculate the critical value
    double alpha = 0.05;  // significance level
    double critical_value = std::sqrt(-0.5 * std::log(alpha / 2.0)) * std::sqrt(static_cast<double>(n));

    bool reject_null = max_diff > critical_value;

    return {max_diff, reject_null};
}

template <typename RNG>
void test_normalcy(){
    RNG rng(42, 0);
    const int num_draws = 200; // should be more than enough

    std::vector<double> sample_data;
    for (int i = 0; i < num_draws/2; i++) 
        sample_data.push_back(rng.template randn<float>());

    while(sample_data.size() < num_draws){
        rnd::double2 d2 = rng.template randn2<double>();
        sample_data.push_back(d2.x);
        sample_data.push_back(d2.y);
    }

    auto [max_diff, reject_null] = ks_test(sample_data);

    std::cout << "Maximum difference between CDFs: " << max_diff << std::endl;
    ASSERT_TRUE(!reject_null);
}

TEST(normal, phillox) {
    test_normalcy<Phillox>();
}

TEST(normal, tyche) {
    test_normalcy<Tyche>();
}

TEST(normal, threefry) {
    test_normalcy<Threefry>();
}

TEST(normal, squares) {
    test_normalcy<Squares>();
}



