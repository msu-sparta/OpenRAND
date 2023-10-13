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
// This compares the raw speed of all the generators on both CPU and GPU.

#include <omp.h>

#include <chrono>
#include <cstdint>
#include <iostream>

#include <openrand/phillox.h>
#include <openrand/squares.h>
#include <openrand/threefry.h>
#include <openrand/tyche.h>

const int N = 268435456;  // no of 32 bits integers required for 1 GB data

template <typename RNG>
double measure_speed() {
  RNG rng(12345, 0);

  auto start = std::chrono::high_resolution_clock::now();

  // `sum`: don't let the compiler throw away entire loop
  uint32_t sum = 0;
  for (int i = 0; i < N; ++i) {
    auto out = rng.template draw<uint32_t>();
    sum += out;
    // don't let the compiler unroll this loop
    if (out == 81) i++;
  }

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  sum &= 1;  // avoid polluting the output

  // Total gigabytes produced
  double total_gb = N * sizeof(uint32_t) / 1e9;
  double time_taken = duration.count() / 1e6;

  // speed: GB/s
  double speed = total_gb / time_taken;

  std::cout << "Speed: " << speed << " GB/s " << sum << std::endl;
  return duration.count();
}

template <typename RNG>
double measure_speed_openmp() {
  uint64_t global_sum = 0;  // Global sum

  auto start = std::chrono::high_resolution_clock::now();

// Parallelize the loop with OpenMP
#pragma omp parallel
  {
    // Each thread gets its own RNG instance
    int thread_id = omp_get_thread_num();
    int local_N = N / omp_get_num_threads();
    // cout<<local_N<<endl;
    uint32_t sum = 0;

    RNG rng(12345 + thread_id, 0);
    for (int i = 0; i < local_N; ++i) {
      auto out = rng.template draw<uint32_t>();
      sum += out;
      // don't let the compiler unroll this loop
      if (out == 81) i++;
    }

#pragma omp critical
    { global_sum += sum; }
  }

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  global_sum &= 1;  // avoid polluting the output

  // Total gigabytes produced
  double total_gb = N * sizeof(uint32_t) / 1e9;
  double time_taken = duration.count() / 1e6;

  // speed: GB/s
  double speed = total_gb / time_taken;

  std::cout << "Speed: " << speed << " GB/s " << global_sum << std::endl;
  return duration.count();
}

int main() {
  std::cout << "====Phillox==== " << std::endl;
  measure_speed<openrand::Phillox>();

  std::cout << "====Threefry====" << std::endl;
  measure_speed<openrand::Threefry>();

  std::cout << "====Squares====" << std::endl;
  measure_speed<openrand::Squares>();

  std::cout << "====Tyche====" << std::endl;
  measure_speed<openrand::Tyche>();

  std::cout << "====Phillox OpenMP====" << std::endl;
  measure_speed_openmp<openrand::Phillox>();

  std::cout << "====Threefry OpenMP====" << std::endl;
  measure_speed_openmp<openrand::Threefry>();

  std::cout << "====Squares OpenMP====" << std::endl;
  measure_speed_openmp<openrand::Squares>();

  std::cout << "====Tyche OpenMP====" << std::endl;
  measure_speed_openmp<openrand::Tyche>();

  return 0;
}
