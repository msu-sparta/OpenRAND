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

#include <openrand/phillox.h>
#include <openrand/squares.h>
#include <openrand/tyche.h>

#include <cstdint>
#include <iostream>
#include <vector>

struct Particle {
  const int global_id;
  int counter = 0;
  double pos[3];

  explicit Particle(int id) : global_id(id) {
  }
};  //  class Particle

int main() {
  using RNG = openrand::Phillox;  // Or, for example, Tyche

  RNG rng(1ULL, 0);

  // Draw random numbers of many types
  int a = rng.rand<int>();
  auto b = rng.rand<long long int>();
  double c = rng.rand<double>();
  float f = rng.rand<float>();

  if constexpr (std::is_same_v<RNG, typename openrand::Phillox>) {
    // this function is not availabe for all generators.
    openrand::float4 f4 = rng.draw_float4();

    std::cout << a << ", " << b << " " << c << " " << f << " " << f4.x << " "
              << f4.y << " " << f4.z << " " << f4.w << std::endl;
  }

  // Create independent streams of numbers in parallel
  float data[16][10];

#pragma omp parallel for
  for (int i = 0; i < 16; i++) {
    RNG rng(i, 0);
    for (int j = 0; j < 10; j++) data[i][j] = rng.rand<float>();
  }

  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 10; j++) std::cout << data[i][j] << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;

  // How to use a unique, independent RNG for each particle in a simulation-
  // The key is to maintain a counter variable for each particle, and
  // increment it each time the rng is instantiated.
  std::vector<Particle> system;
  for (int i = 0; i < 16; i++) system.emplace_back(i);

// initialize
#pragma omp parallel for
  for (int i = 0; i < 16; i++) {
    Particle &p = system[i];
    // If you don't increment p.counter here, you're going to get exactly
    // the same values in the next loop.
    RNG rng1(p.global_id, p.counter++);
    for (int j = 0; j < 3; j++) p.pos[j] = rng1.rand<double>();
  }

// a random step
#pragma omp parallel for
  for (int i = 0; i < 16; i++) {
    Particle &p = system[i];
    RNG rng2(p.global_id, p.counter++);
    for (int j = 0; j < 3; j++) p.pos[j] += rng2.rand<double>() / 10;
  }

  for (int i = 0; i < 16; i++) {
    Particle &p = system[i];
    for (int j = 0; j < 3; j++) std::cout << p.pos[j] << " ";
    std::cout << p.counter << std::endl;
  }

  return 0;
}
