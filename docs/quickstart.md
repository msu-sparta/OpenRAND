# A Quick Introduction to OpenRAND

OpenRAND comes with four generator classes: `Phillox`, `Tyche`, `Threefry` and `Squares`. They all have a similar interface: given a `seed` and `counter`, the generator can produce a stream of random numbers (upto 2^32 numbers per object). `seed` should be self-explanatory, we'll introduce `counter` later.

```
#include <phillox.h>

int main() {
    using RNG = Phillox; // You can swap with Tyche, Threefry or Squares
    
    // Initialize RNG with seed and counter
    RNG rng(/*seed*/ 42, /*counter*/ 0);

    for(int i=0; i<(2<<31); i++){
        // Draw an integer from uniform distribution
        int a = rng.rand<int>();
    }
    ...
}
```

You can also draw random numbers of other types, such as `long long int`, `double`, `float`, etc.

```
    ...
    for(int i=0; i<(2<<31); i++){
        auto b = rng.rand<long long int>();
        double c = rng.rand<double>();
        float f = rng.rand<float>();
    }
    ...
```

The API follows numpy style. If you want a floating point from a normal distribution, use `randn`:

```
    ...
    auto f = rng.randn<float>();
    
    // A double with mean 10 and standard deviation 5
    auto d = rng.randn<double>(10.0, 5.0);
```

## Parallel Streams

In a parellel program, it's a good idea to let all your threads have their local random streams. In an OpenMP program you can - 

```
#pragma omp parallel
{
    int seed = omp_get_thread_num();
    RNG gen(seed, 0);

    int hits = 0;
    for (int i = 0; i < nsamples; i++) {
        float x = gen.rand(); // default template parameter is float in range [0,1).
        ...
    }
}
```
Make sure the seeds are unique, OpenRAND will ensure these streams will not have any correlation between them. This is an important property of OpenRAND that sets it apart from many libraries out there.

Another thing to remember is that these generator objects are very cheap to create, destroy and have a very small memory footprint. So don't worry about creating a new generator object for each thread even if you've got a million threads. Which brings us to-

## GPU
It's time to introduce the `counter` argument. 

You can think of a single `seed` capable of producing a unique stream of length 2^64. This stream is divided into 2^32 *substreams*, each of length 2^32. The `counter` argument is used to select a substream. 

For reproducibility, on GPU, it's better to think in terms of work-unit or processing element instead of thread. For example, in a history-based simulation, a work-unit models the entire lifespan of one particle. In a ray tracing renderer, it can be a pixel index. A work-unit can undergo multiple kerel launches in it's lifetime. 

One random number stream par particle isn't ideal, it's good to have one at each kernel launch for a particle. In this way, you can avoid the overhead of a seperate kernel launch just to initialize random states, loading them from global memory inside each kernel, and saving back the modified state etc. `counter` helps you get around all that, by cheaply creating a substream for each kernel launch.


Here's an example of a typical use-case on GPU: a monte carlo paticle simulation code that runs for 10,000 time steps. We can simply use the iteration number as `counter`. For `seed`, we assume each thread below has a unique global id atribute called `pid`. 

```
__global__ 
void apply_forces(Particle *particles, int counter){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Particle p = particles[i];
    ...

    // create thread-local rng object
    RNG local_rand_state(p.pid, counter);
    
    p.vx += (local_rand_state.rand<double>()  * 2.0 - 1.0);
    ...
}


int main(){
    ...

    // Simulation loop
    int iter = 0;
    while (iter++ < STEPS) {
        apply_forces<<<nblocks, nthreads>>>(particles, iter);
        ...
    }
}

```



