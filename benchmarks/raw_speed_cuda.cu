#include <iostream>
#include <chrono>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda.h>


#include "phillox.h"
#include "threefry.h"
#include "squares.h"
#include "tyche.h"

using std::cout;
using std::endl;
using namespace std::chrono;

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
    auto start = high_resolution_clock::now();

    measure_speed_cuda_kernel<RNG><<<numBlocks, numThreadsPerBlock>>> \
        (global_sum_dev, nums_per_thread);
    cudaDeviceSynchronize();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

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


    cout<<"Phillox: "<<endl;
    measure_speed_cuda<Phillox>(deviceProp.multiProcessorCount);

    cout<<"Threefry: "<<endl;
    measure_speed_cuda<Threefry>(deviceProp.multiProcessorCount);

    cout<<"Squares: "<<endl;
    measure_speed_cuda<Squares>(deviceProp.multiProcessorCount);

    cout<<"Tyche: "<<endl;
    measure_speed_cuda<Tyche>(deviceProp.multiProcessorCount);

    return 0;
}