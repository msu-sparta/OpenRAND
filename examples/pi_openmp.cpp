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

#include <omp.h>

#include <atomic>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <openrand/phillox.h>
#include <openrand/tyche.h>

double compute_pi() {
  using RNG = openrand::Phillox;
  const int nsamples = 10000000;
  int total_samples;
  int total_hits = 0;

#pragma omp parallel
  {
    total_samples = nsamples * omp_get_num_threads();

    int seed = omp_get_thread_num();
    RNG gen(seed, 0);

    int hits = 0;
    for (int i = 0; i < nsamples; i++) {
      float x = gen.rand();
      float y = gen.rand();

      if (x * x + y * y <= 1.0) hits++;
    }

#pragma omp atomic
    total_hits += hits;
  }

  double pi_estimate = 4.0 * total_hits / total_samples;
  return pi_estimate;
}

int main() {
  double pi_estimate = compute_pi();

  constexpr double pi = 3.14159265358979323846;

  std::cout << "pi_estimate: " << pi_estimate << std::endl;
  std::cout << "log10(|pi - pi_estimate|): " << std::log10(std::abs(pi - pi_estimate)) << std::endl;

  return 0;
}
