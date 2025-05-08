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

#include <openrand/philox.h>
#include <openrand/squares.h>
#include <openrand/threefry.h>
#include <openrand/tyche.h>

#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>

constexpr int N = 268435456;
constexpr int BLOCKSIZE = 256;

template <typename RNG>
double measure_speed_sycl(sycl::queue& q, int num_blocks, bool warmup = false) {
  int total_threads = num_blocks * BLOCKSIZE;
  int nums_per_thread = N / total_threads;

  // Output buffer for per-block sums
  std::vector<uint32_t> global_sum(num_blocks, 0);
  {
    sycl::buffer<uint32_t> sum_buf(global_sum.data(),
                                   sycl::range<1>(num_blocks));

    auto start = std::chrono::high_resolution_clock::now();

    q.submit([&](sycl::handler& h) {
      auto sum_acc = sum_buf.get_access<sycl::access::mode::write>(h);

      // FIXED: Allocate shared (local) memory for each block
      sycl::local_accessor<uint32_t, 1> temp_storage(sycl::range<1>(BLOCKSIZE),
                                                     h);

      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_threads),
                                       sycl::range<1>(BLOCKSIZE)),
                     [=](sycl::nd_item<1> item) {
                       int tid = item.get_global_id(0);
                       int lid = item.get_local_id(0);
                       int bid = item.get_group(0);

                       uint32_t local_sum = 0;
                       RNG rng(12345 + tid, 0);

                       for (int i = 0; i < nums_per_thread; ++i) {
                         uint32_t tmp = rng.template draw<uint32_t>();
                         local_sum += tmp;
                         if (tmp == 81) i++;
                       }

                       temp_storage[lid] = local_sum;
                       item.barrier(sycl::access::fence_space::local_space);

                       for (int stride = BLOCKSIZE / 2; stride > 0;
                            stride /= 2) {
                         if (lid < stride)
                           temp_storage[lid] += temp_storage[lid + stride];
                         item.barrier(sycl::access::fence_space::local_space);
                       }

                       if (lid == 0) sum_acc[bid] = temp_storage[0];
                     });
    });

    q.wait_and_throw();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    double total_gb = N * sizeof(uint32_t) / 1e9;
    double time_taken = duration.count() / 1e6;
    double speed = total_gb / time_taken;

    if (!warmup)
      std::cout << "Speed: " << speed << " GB/s  (sum = " << global_sum[0]
                << ")\n";

    return duration.count();
  }
}

int main() {
  sycl::queue q(sycl::gpu_selector_v);
  auto dev = q.get_device();
  int num_SMs = dev.get_info<sycl::info::device::max_compute_units>();

  std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << "\n";
  std::cout << "Number of Compute Units (SMs): " << num_SMs << "\n";

  std::cout << "====Philox====" << std::endl;
  measure_speed_sycl<openrand::Philox>(q, num_SMs * 8);

  std::cout << "====Threefry====" << std::endl;
  measure_speed_sycl<openrand::Threefry>(q, num_SMs * 8);

  std::cout << "====Squares====" << std::endl;
  measure_speed_sycl<openrand::Squares>(q, num_SMs * 8);

  std::cout << "====Tyche====" << std::endl;
  measure_speed_sycl<openrand::Tyche>(q, num_SMs * 8);

  return 0;
}
