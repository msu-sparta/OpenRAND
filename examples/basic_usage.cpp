#include <cstdint>
#include <iostream>
#include <vector>

#include "phillox.h"
#include "tyche.h"

using std::cout;
using std::endl;

struct Particle {
  const int global_id;
  int counter = 0;
  double pos[3];

  Particle(int id) : global_id(id) {}
};

int main() {
  using RNG = Phillox; // Tyche

  RNG rng(1, 0);

  // Draw random numbers of many types
  int a = rng.rand<int>();
  auto b = rng.rand<long long int>();
  double c = rng.rand<double>();
  float f = rng.rand<float>();

  rnd::float4 f4 = rng.draw_float4();

  cout << a << ", " << b << " " << c << " " << f << " " << f4.x << " " << f4.y
       << " " << f4.z << " " << f4.w << endl;

  // Create independent streams of numbers in parallel
  float data[16][10];

#pragma omp parallel for
  for (int i = 0; i < 16; i++) {
    Phillox rng(i, 0);
    for (int j = 0; j < 10; j++)
      data[i][j] = rng.rand<float>();
  }

  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 10; j++)
      cout << data[i][j] << " ";
    cout << endl;
  }
  cout << endl;

  // How to use a unique, independent RNG for each particle in a simulation-
  // The key is to maintain a counter variable for each particle, and
  // increment it each time the rng is instantiated.
  std::vector<Particle> system;
  for (int i = 0; i < 16; i++)
    system.emplace_back(i);

// initialize
#pragma omp parallel for
  for (int i = 0; i < 16; i++) {
    Particle &p = system[i];
    // if you don't increment p.counter here, you're going to get exactly
    // same values in the next loop.
    Phillox rng1(p.global_id, p.counter++);
    for (int j = 0; j < 3; j++)
      p.pos[j] = rng1.rand<double>();
  }

// a random step
#pragma omp parallel for
  for (int i = 0; i < 16; i++) {
    Particle &p = system[i];
    Phillox rng2(p.global_id, p.counter++);
    for (int j = 0; j < 3; j++)
      p.pos[j] += rng2.rand<double>() / 10;
  }

  for (int i = 0; i < 16; i++) {
    Particle &p = system[i];
    for (int j = 0; j < 3; j++)
      cout << p.pos[j] << " ";
    cout << p.counter << endl;
  }

  return 0;
}
