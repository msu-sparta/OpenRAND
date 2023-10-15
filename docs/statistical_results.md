# Statistical Quality Test Results

OpenRAND generators has been tested with two statistical suites: TestU01 and PractRand. 

There are two ways to test the generators:
+ Single stream: we use PractRand to test a single random number stream to their theoratical limit of 2^32 numbers (2^34 bytes). See the files [practrand_[tyche|squares|threefry|phillox].txt](https://github.com/msu-sparta/OpenRAND/tree/gh-pages/results) for details. 

+ Parallel Streams: This tests for correlation among parallel streams. This is not quite straightforward in those statistical frameworks- as they are designed to work on a single stream. To stitch parallel streams to a single one, we use the following approach: 
  + Start with a set of `N` random number generators.
  + Generate `C` random numbers from each generator.
  + Combine them, now we have a buffer of length `N*C`.
  + Repeat the above steps to extend the buffer by `N*C` at each iteration as long as necessary.

In this document, our primary focus is on parallel streams, as single streams are already well-examined in existing literature. For the results discussed here, we've set N at 100 and C at 3.

## Practrand
Practrand can consume practically infinite amount of data. Given that each stream contains 2^32 numbers, and each number is 4 bytes (2^2), with 100 streams (or 2^6.64), the combined buffer is little over 2^40 bytes long. Put another way, we shouldn't expect to see any failures in Practrand before reaching the 2^40 bytes (1-terabyte) mark.

None of the generators failed this test. For detailed output from the practrand suite for for each generator, see: [practrandm_[tyche|squares|threefry|phillox].txt](https://github.com/msu-sparta/OpenRAND/tree/gh-pages/results).


## TestU01
Testu01 has 3 batteries of tests: SmallCrush, Crush, and BigCrush. BigCrush is the most comprehensive one with a total of 106 tests, and is the one we use.

Some generators do fail one or two BigCRush tests. For example, this is the output from Philox:
```
========= Summary results of BigCrush =========

 Version:          TestU01 1.2.3
 Generator:        philox
 Number of statistics:  160
 Total CPU time:   04:36:59.22
 The following tests gave p-values outside [0.001, 0.9990]:
 (eps  means a value < 1.0e-300):
 (eps1 means a value < 1.0e-15):

       Test                          p-value
 ----------------------------------------------
  9  CollisionOver, t = 14           0.9994 
 ----------------------------------------------
 All other tests were passed
```

This is not unusual on some runs, authors of curand also [observed](https://docs.nvidia.com/cuda/curand/testing.html) some failures on Bigcrush. It also depends on the configuration we used. 

For detailed output from Bigcrush suite, see [testu01m_[tyche|squares|threefry|phillox].txt](https://github.com/msu-sparta/OpenRAND/tree/gh-pages/results). 