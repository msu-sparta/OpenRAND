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

#include <benchmark/benchmark.h>
#include <openrand/philox.h>
#include <openrand/squares.h>
#include <openrand/threefry.h>
#include <openrand/tyche.h>

#include <random>

template <typename RNG>
static void bench_rng(benchmark::State& state) {
  for (auto _ : state) {
    RNG rng(12345, 0);
    for (int i = 0; i < state.range(0); ++i)
      benchmark::DoNotOptimize(rng.template draw<uint32_t>());
  }
}

template <typename RNG>
static void bench_cpp(benchmark::State& state) {
  for (auto _ : state) {
    RNG rng(12345);
    for (int i = 0; i < state.range(0); ++i) benchmark::DoNotOptimize(rng());
  }
  //benchmark::ClobberMemory();
}

#define TUnit benchmark::kNanosecond

// Register the function as a benchmark
BENCHMARK(bench_rng<openrand::Philox>)
    ->Unit(TUnit)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);
BENCHMARK(bench_rng<openrand::Tyche>)
    ->Unit(TUnit)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);
BENCHMARK(bench_rng<openrand::Squares>)
    ->Unit(TUnit)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);
BENCHMARK(bench_rng<openrand::Threefry>)
    ->Unit(TUnit)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);

BENCHMARK(bench_cpp<std::mt19937>)
    ->Unit(TUnit)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);

BENCHMARK_MAIN();