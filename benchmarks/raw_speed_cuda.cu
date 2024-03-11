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

#include <iostream>
#include <chrono>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda.h>

#include <openrand/philox.h>
#include <openrand/threefry.h>
#include <openrand/squares.h>
#include <openrand/tyche.h>

const int N = 268435456; // no of 32 bits integers required for 1 GB data
const int BLOCKSIZE = 256;

template<typename RNG>
__global__ void measure_speed_cuda_kernel(uint32_t *global_sum_dev, int num_per_thread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t sum = 0;

    RNG rng(12345 + idx, 0);

    for (int i = 0; i < num_per_thread; i ++) {
        auto tmp = rng.template draw<uint32_t>();
        sum += tmp;
        if(tmp==81) i++;
    }


    // All these to make sure `sum` doesn't get optimized away
    typedef cub::BlockReduce<uint32_t, BLOCKSIZE> BlockReduceT; 
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    uint32_t result;
    if(idx < N) result = BlockReduceT(temp_storage).Sum(sum);

    if(threadIdx.x == 0) {
        global_sum_dev[blockIdx.x] = result;   
    }
}


template<typename RNG>
double measure_speed_cuda(int numSMs, bool warmup=false) {
    // Launch configuration doesn't imitate random123
    int numBlocks = numSMs * 8;
    int numThreadsPerBlock = BLOCKSIZE;

    uint32_t *global_sum;
    uint32_t *global_sum_dev;

    // Allocate memory for global_sum on the device
    global_sum = (uint32_t*) malloc(sizeof(uint32_t) * numBlocks);
    cudaMalloc((void **)&global_sum_dev, sizeof(uint32_t) * numBlocks);

    int nums_per_thread = N / (numBlocks * numThreadsPerBlock);
    auto start = std::chrono::high_resolution_clock::now();

    measure_speed_cuda_kernel<RNG><<<numBlocks, numThreadsPerBlock>>> \
        (global_sum_dev, nums_per_thread);
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    cudaMemcpy(global_sum, global_sum_dev, sizeof(uint32_t)*numBlocks, cudaMemcpyDeviceToHost);
    cudaFree(global_sum_dev);
    free(global_sum);


    // Total gigabytes produced
    double total_gb = N * sizeof(uint32_t) / 1e9;
    double time_taken = duration.count() / 1e6;

    // Speed: GB/s
    double speed = total_gb / time_taken;


    if(!warmup) std::cout << "Speed: " << speed << " GB/s " << global_sum[0] << std::endl;
    return duration.count();
}

int main(){
    cudaDeviceProp deviceProp;
    int device;
    cudaGetDevice(&device); // Get current device
    cudaGetDeviceProperties(&deviceProp, device);

    std::cout << "Number of Streaming Multiprocessors (SMs): " << deviceProp.multiProcessorCount << std::endl;


    std::cout << "====Philox====" << std::endl;
    measure_speed_cuda<openrand::Philox>(deviceProp.multiProcessorCount);

    std::cout << "====Threefry====" << std::endl;
    measure_speed_cuda<openrand::Threefry>(deviceProp.multiProcessorCount);

    std::cout << "====Squares====" << std::endl;
    measure_speed_cuda<openrand::Squares>(deviceProp.multiProcessorCount);

    std::cout << "====Tyche====" << std::endl;
    measure_speed_cuda<openrand::Tyche>(deviceProp.multiProcessorCount);

    return 0;
}
