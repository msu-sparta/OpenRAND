#include <gtest/gtest.h>
#include "phillox.h"
#include "tyche.h"
#include "threefry.h"
#include "squares.h"

using std::cout;
using std::endl;

// TODO: Add std::is_trivially_copyable and std::is_trivially_destructible tests for all generator types


template <typename RNG>
void test_basic(){
    RNG rng(42, 0);
    EXPECT_NE(rng.template rand<long long int>(), rng.template rand<long long int>());

    RNG rng1(1, 0);
    RNG rng2(1, 0);
    EXPECT_EQ(rng1.template rand<int>(), rng2.template rand<int>());
    EXPECT_EQ(rng1.template rand<int>(), rng2.template rand<int>());

    RNG rng3(3, 0);
    RNG rng4(4, 0);
    EXPECT_NE(rng3.template rand<int>(), rng4.template rand<int>()); // this "could" happen :)
}

TEST(RNG, basic) {
    test_basic<Phillox>();
    test_basic<Tyche>();
    test_basic<Threefry>();
    test_basic<Squares>();
}

template <typename RNG>
void test_mean(){
    RNG rng(0, 0);
    int num_draws = 1000;

    float mean = 0;
    for (int i = 0; i < num_draws; i++) {
        mean += rng.template rand<float>();
    }
    mean /= num_draws;
    EXPECT_NEAR(mean, 0.5, 0.0103); // 99.99 % confidence
}

TEST(Uniform, mean) {
    test_mean<Phillox>();
    test_mean<Tyche>();
    test_mean<Threefry>();
    test_mean<Squares>();
}

