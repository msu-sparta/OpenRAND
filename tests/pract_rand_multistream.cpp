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

#include <openrand/phillox.h>
#include <openrand/squares.h>
#include <openrand/threefry.h>
#include <openrand/tyche.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Control parameters for this test program.
const int C = 3;     // C numbers from per stream
const int NS = 100;  // from NS streams
// end

template <typename RNG>
void init_generators(std::vector<RNG> *const generators_ptr, const int ctr) {
  for (int i = 0; i < NS; i++) {
    generators_ptr->emplace_back(i, ctr);
  }
}

template <typename RNG>
void populate_buffer(std::vector<RNG> *const generators_ptr,
                     std::vector<uint32_t> *const buffer_ptr) {
  auto generators = *generators_ptr;
  for (int i = 0; i < NS; i++) {
    for (int j = 0; j < C; j++) {
      buffer_ptr->push_back(generators[i].template draw<uint32_t>());
    }
  }
  std::reverse(buffer_ptr->begin(), buffer_ptr->end());
}

template <typename RNG>
void generate_multi_stream() {
  std::vector<RNG> generators;
  std::vector<uint32_t> buffer;

  for (int ctr = 0;; ctr++) {
    init_generators(&generators, ctr);
    populate_buffer(&generators, &buffer);
    fwrite((void *)buffer.data(), sizeof(buffer[0]), NS * C, stdout);

    // clear both vectors
    buffer.clear();
    generators.clear();
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout
        << "Please provide one argument specifying the generator name.\n"
        << "Valid options are 'philox', 'tyche', 'threefry', or 'squares'."
        << std::endl;
    return 1;
  }

  std::string arg = argv[1];

  std::cout << "Generating multi-stream for: " << arg << std::endl;

  if (arg == "philox") {
    generate_multi_stream<openrand::Phillox>();
  } else if (arg == "tyche") {
    generate_multi_stream<openrand::Tyche>();
  } else if (arg == "threefry") {
    generate_multi_stream<openrand::Threefry>();
  } else if (arg == "squares") {
    generate_multi_stream<openrand::Squares>();
  } else {
    std::cout << "Invalid argument." << std::endl;
  }
  return 0;
}
