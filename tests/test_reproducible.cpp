#include <cstdio>
#include <random>
#include <openrand/philox.h>

int main() {
    using RNG = openrand::Philox;  // Or, for example, Tyche

    // Initialize RNG with seed and counter
    RNG rng(1, 0);

    // Draw uniform random numbers of many types
    int a = rng.rand<int>();            // range [0,2^32)
    auto b = rng.rand<long long int>(); // range [0,2^64)
    double c = rng.rand<double>();      // range [0,1)
    float f = rng.rand<float>();        // range [0,1)

    // OpenRAND's own API: this sequence should match across platforms.
    for(int i=0; i<5; i++)
    {
        double x = rng.rand<double>();
        printf("openrand x = %.17g\n", x);
    }

    // std distribution functions are implemented by the C++ standard library.
    // These values may differ across libstdc++, libc++, etc.
    RNG std_rng(1, 0);
    std_rng.rand<int>();
    std_rng.rand<long long int>();
    std_rng.rand<double>();
    std_rng.rand<float>();

    std::lognormal_distribution<double> dist(0.0, 1.0);
    for(int i=0; i<5; i++)
    {
        double x = dist(std_rng);
        printf("std x = %.17g\n", x);
    }

    (void)a;
    (void)b;
    (void)c;
    (void)f;
}
