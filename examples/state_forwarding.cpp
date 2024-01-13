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
 * This example demonstrates the use of the forward_state function to generate
 * output identical to that of serial code, even when using a variable number
 * of threads.
 */

#include <openrand/philox.h>
#include <openrand/squares.h>
#include <openrand/threefry.h>

#include <iostream>
#include <vector>

int main() {
  // all generators except Tyche support forward_state
  using RNG = openrand::Threefry;
  RNG master_rng(42, 0);

  // create multiple rngs by forwarding master's state
  std::vector<RNG> rngs;
  for (int i = 0; i < 3; i++) {
    rngs.emplace_back(master_rng.forward_state(5 * i));
  }

  // the serial output from master_rng
  std::cout << "master_rng output:" << std::endl;
  for (int i = 0; i < 15; i++) std::cout << master_rng.rand() << " ";
  std::cout << std::endl;

  // the output from parallel rngs
  float vals[15];

#pragma omp parallel for
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) vals[5 * i + j] = rngs[i].rand();
  }

  std::cout << "parallel rng output:" << std::endl;
  for (int i = 0; i < 15; i++) std::cout << vals[i] << " ";

  std::cout << std::endl;
  return 0;
}