#include <atomic>
#include <cstdint>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

#include "../include/phillox.h"
#include "../include/tyche.h"

using std::cout;
using std::endl;


double compute_pi(){
  using RNG = Phillox;
  const int nsamples = 10000000;
  int total_samples;
  int total_hits = 0;

#pragma omp parallel
  {
    total_samples = nsamples * omp_get_num_threads();

    int seed = omp_get_thread_num();
    RNG gen(seed, 0);

    int hits = 0;
    for (int i = 0; i < nsamples; i++) {
      float x = gen.rand();
      float y = gen.rand();

      if (x * x + y * y <= 1.0)
        hits++;
    }

#pragma omp atomic
    total_hits += hits;
  }

  double pi_estimate = 4.0 * total_hits / total_samples;
  return pi_estimate;
}


int main() {

  double pi_estimate = compute_pi();

  cout << pi_estimate << endl;

  return 0;
}
