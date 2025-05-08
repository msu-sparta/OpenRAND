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

/**
 * Compute Pi using monte carlo method.
 *
 * For simplicity, we ignore usual error checking here.
 */

#include <openrand/philox.h>  // Make sure this works in SYCL

#include <cmath>
#include <iostream>
#include <sycl/sycl.hpp>

constexpr int N = 100'000'000;                    // Total samples
constexpr int SAMPLES_PER_THREAD = 1000;          // Per-thread work
constexpr int NTHREADS = N / SAMPLES_PER_THREAD;  // Number of SYCL work-items
constexpr int THREADS_PER_BLOCK = 256;            // Suggestion only

using RNG = openrand::Philox;

int main() {
  sycl::queue q{sycl::default_selector_v};

  std::cout << "Number of samples: " << N << std::endl;
  std::cout << "Samples per thread: " << SAMPLES_PER_THREAD << std::endl;
  std::cout << "Number of threads: " << NTHREADS << std::endl;

  int h_sum = 0;

  {
    sycl::buffer<int> sum_buf(&h_sum, 1);

    q.submit([&](sycl::handler &h) {
      auto sum_acc = sum_buf.get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(sycl::range<1>{NTHREADS}, [=](sycl::id<1> idx) {
        int localHits = 0;
        RNG rng(idx[0], 0);

        for (int i = 0; i < SAMPLES_PER_THREAD; ++i) {
          float x = rng.rand();
          float y = rng.rand();
          if (x * x + y * y <= 1.0f) ++localHits;
        }

        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_sum(sum_acc[0]);

        atomic_sum.fetch_add(localHits);
      });
    });
  }

  double pi_estimate = 4.0 * static_cast<double>(h_sum) / N;
  constexpr double pi = 3.14159265358979323846;
  std::cout << "pi_estimate: " << pi_estimate << std::endl;
  std::cout << "log10(|pi - pi_estimate|): "
            << std::log10(std::abs(pi - pi_estimate)) << std::endl;

  return 0;
}
