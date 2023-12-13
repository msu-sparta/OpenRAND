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

#include <openrand/philox.h>
#include <openrand/squares.h>
#include <openrand/threefry.h>
#include <openrand/tyche.h>

#include <cstdint>
#include <cstdio>

// From build directory, run this as (modify path and generator name as needed):
// ./tests/pract_rand philox | /home/shihab/codes/PractRand/RNG_test stdin32

template <typename RNG>
void generate_stream() {
  for (int ctr = 0;; ctr++) {
    RNG rng(0, ctr);
    for (int i = 0; i < 10; i++) {
      uint32_t value = rng.template draw<uint32_t>();
      fwrite((void*)&value, sizeof(value), 1, stdout);
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout
        << "Please provide one argument specifying the generator name.\n"
        << "Valid options are 'philox', 'tyche', 'threefry', or 'squares'."
        << std::endl;
    return 1;
  }

  std::string arg = argv[1];

  std::cout << "Generating stream for: " << arg << std::endl;

  if (arg == "philox") {
    generate_stream<openrand::Philox>();
  } else if (arg == "tyche") {
    generate_stream<openrand::Tyche>();
  } else if (arg == "threefry") {
    generate_stream<openrand::Threefry>();
  } else if (arg == "squares") {
    generate_stream<openrand::Squares>();
  } else {
    std::cout << "Invalid argument." << std::endl;
  }
  return 0;
}
