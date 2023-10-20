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

#include <curand_kernel.h>
#include <openrand/tyche.h>

#include <cmath>
#include <iostream>

const int N = 100000000;                      // Number of points
const int SAMPLES_PER_THREAD = 1000;          // Number of samples per thread
const int NTHREADS = N / SAMPLES_PER_THREAD;  // Number of threads
const int THREADS_PER_BLOCK = 256;            // Number of threads per block

typedef openrand::Tyche RNG;

__global__ void monteCarloPi(int *d_sum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  RNG rng(idx, 0);
  int localHits = 0;

  for (int i = 0; i < SAMPLES_PER_THREAD; i++) {
    // Generate random numbers in [0, 1]
    float x = rng.rand();
    float y = rng.rand();
    if (x * x + y * y <= 1.0f) localHits++;
  }

  atomicAdd(d_sum, localHits);
}

int main() {
  int *d_sum;

  std::cout << "Number of samples: " << N << std::endl;
  std::cout << "Number of samples per thread: " << SAMPLES_PER_THREAD
            << std::endl;
  std::cout << "Number of threads: " << NTHREADS << std::endl;

  cudaMalloc(&d_sum, sizeof(int));
  cudaMemset(d_sum, 0, sizeof(int));

  int nblocks = (NTHREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  monteCarloPi<<<THREADS_PER_BLOCK, nblocks>>>(d_sum);

  int h_sum;
  cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

  float pi = 4.0 * (float)h_sum / N;

  std::cout << "Approximated value of Pi: " << pi << std::endl;

  cudaFree(d_sum);

  return 0;
}
